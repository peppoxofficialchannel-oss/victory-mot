//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by the translator
//! that translated it (its source fingerprint and settings, see `build`) and
//! by the host facts it was translated for (widest wave, view heap limit), so
//! a changed translator or a different device never serves an old translation.
//! Offline warm-up (`victory translate --cache`) and title execution share the
//! same cache; entries go through `victory-pipeline`'s content-addressed cache,
//! which writes them atomically; an entry that does not read back is
//! translated again.
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too (entries of older builds
//! are ignored anyway).

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use victory_pipeline::{Cache, CacheKey};
use victory_shader::emit::{HostShader, StaticSampler, UserData};
use victory_shader::resource::RootIndexedView;
use victory_shader::{HwStage, RootType, ShaderError};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) -> bool {
    match Cache::open(dir) {
        Ok(cache) => CACHE.set(cache).is_ok(),
        Err(e) => {
            tracing::warn!(error = %e, "no shader cache");
            false
        }
    }
}

/// Select the same hardware and shader capabilities that a title run uses.
pub fn prepare_host() -> bool {
    if crate::host::host().is_none() {
        return false;
    }
    crate::descriptor::host::probe_view_limit();
    true
}

/// The result of looking up or translating one console shader.
pub enum PrecompileOutcome {
    Cached,
    Translated,
    CachedRefusal,
    Refused,
}

/// Fills the runtime shader cache and reports whether the entry was already there.
pub fn precompile_shader(container: &[u8]) -> Result<PrecompileOutcome, String> {
    let cache = CACHE.get().ok_or("no shader cache")?;
    let (result, status) = lookup(cache, container);
    match (result, status) {
        (_, CacheStatus::Failed(error)) => Err(error),
        (Ok(_), CacheStatus::Hit) => Ok(PrecompileOutcome::Cached),
        (Ok(_), CacheStatus::Stored) => Ok(PrecompileOutcome::Translated),
        (Err(_), CacheStatus::Hit) => Ok(PrecompileOutcome::CachedRefusal),
        (Err(_), _) => Ok(PrecompileOutcome::Refused),
        (Ok(_), CacheStatus::Unstored) => Err("translation was not stored".into()),
    }
}

/// Successful shader translations found on disk and generated since startup.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What identifies the translation, besides the container: the shader
/// crate's fingerprint (its source, every switch it reads, the host facts the
/// device gave it, the loop limit, barrier waits and clean-up passes) and
/// this file's encoding of the result. Not the whole executable any more: a
/// rebuild that leaves the translator alone keeps its translations (each new
/// build used to translate the title's thousands of shaders again, which kept
/// the first minutes of every test run short of pipelines).
/// `VICTORY_SHADER_CACHE_KEY=exe` restores the old key (the executable's hash).
/// Worked out per lookup, so a translation made before the device said its
/// wave size is never filed under the later key.
fn build() -> [u8; 32] {
    static WARNED: std::sync::Once = std::sync::Once::new();
    let limits = victory_shader::emit::loop_limits();
    if limits != victory_shader::emit::DEFAULT_LOOP_LIMITS {
        WARNED.call_once(|| {
            tracing::warn!(?limits, "VICTORY_SHADER_LOOP_LIMIT: translated loops stop elsewhere")
        });
    }
    static EXE: OnceLock<Option<[u8; 32]>> = OnceLock::new();
    let exe = EXE.get_or_init(|| {
        if std::env::var_os("VICTORY_SHADER_CACHE_KEY").is_some_and(|v| v == "exe") {
            match std::env::current_exe().and_then(std::fs::read) {
                Ok(bytes) => Some(*blake3::hash(&bytes).as_bytes()),
                Err(e) => {
                    tracing::warn!(error = %e, "cannot read our executable; keying on the translator");
                    None
                }
            }
        } else {
            None
        }
    });
    let mut h = blake3::Hasher::new();
    h.update(b"victory shader cache key 2\0");
    if let Some(exe) = exe {
        h.update(exe);
    }
    h.update(&victory_shader::emit::translation_fingerprint());
    h.update(include_bytes!("shader_cache.rs"));
    *h.finalize().as_bytes()
}

/// The offline warm-up (`victory translate --warm`): `container` is
/// translated into the cache unless it is there already. `Ok(true)`: it
/// was translated now. Needs [`use_shader_cache`] and the same host facts
/// the run will have (`victory_shader::emit::translation_fingerprint`).
pub fn warm(container: &[u8]) -> Result<bool, String> {
    let cache = CACHE.get().ok_or("no shader cache")?;
    let ran = AtomicBool::new(false);
    let translate = |c: &[u8]| {
        ran.store(true, Ordering::Relaxed);
        victory_shader::emit::translate(c)
    };
    let (result, status) = cached(
        cache,
        &build(),
        target(),
        canonical_container(container),
        translate,
        || true,
    );
    match status {
        CacheStatus::Failed(error) => Err(error),
        _ => result.map(|_| ran.load(Ordering::Relaxed)),
    }
}

/// Uses an in-memory translation when the disk cache cannot be written.
pub(crate) fn host_shader(container: &[u8]) -> Result<HostShader, String> {
    let (result, status) = match CACHE.get() {
        Some(cache) => lookup(cache, container),
        None => (
            victory_shader::emit::translate(container).map_err(|e| e.to_string()),
            CacheStatus::Unstored,
        ),
    };
    if let CacheStatus::Failed(error) = &status {
        static WARNED: OnceLock<()> = OnceLock::new();
        WARNED.get_or_init(|| tracing::warn!(%error, "shader cache unavailable for an entry"));
    }
    if result.is_ok() {
        let counter = if status == CacheStatus::Hit {
            &HITS
        } else {
            &MISSES
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }
    result
}

fn target() -> [u32; 2] {
    [
        victory_shader::emit::host_max_wave(),
        victory_shader::emit::view_heap_limit(),
    ]
}

/// The translation of a console compute program for dispatches by thread
/// counts whose last groups are cut short
/// (`victory_shader::emit::translate_thread_limited`), from the cache when
/// there is one, under a key of its own.
pub(crate) fn host_shader_thread_limited(container: &[u8]) -> Result<HostShader, String> {
    let container = canonical_container(container);
    let mut result = None;
    for _ in 0..2 {
        let expected = target();
        let build = build();
        let (translated, _status) = match CACHE.get() {
            Some(cache) => cached_for(
                cache,
                &build,
                expected,
                THREAD_COUNTS,
                container,
                victory_shader::emit::translate_thread_limited,
                || target() == expected,
            ),
            None => (
                victory_shader::emit::translate_thread_limited(container).map_err(|e| e.to_string()),
                CacheStatus::Unstored,
            ),
        };
        result = Some(translated);
        if target() == expected {
            break;
        }
    }
    result.unwrap_or_else(|| {
        Err("host shader capabilities changed during translation".to_owned())
    })
}

/// A cache key's share for a translation of the same container another way
/// ([`host_shader_thread_limited`]).
const THREAD_COUNTS: &[u8] = b"thread counts\0";

/// DXBC's declared size excludes trailing allocation padding. Keep the title's
/// original bytes for pipeline identity, but use this slice for translation.
fn canonical_container(bytes: &[u8]) -> &[u8] {
    if bytes.starts_with(b"DXBC")
        && let Some(size) = bytes.get(0x18..0x1c)
    {
        let size = u32::from_le_bytes(size.try_into().expect("four bytes")) as usize;
        if size >= 0x20
            && let Some(container) = bytes.get(..size)
        {
            return container;
        }
    }
    bytes
}

#[derive(Debug, PartialEq, Eq)]
enum CacheStatus {
    Hit,
    Stored,
    Unstored,
    Failed(String),
}

type Translation = Result<HostShader, String>;

fn lookup(cache: &Cache, container: &[u8]) -> (Translation, CacheStatus) {
    let container = canonical_container(container);
    for _ in 0..2 {
        let expected = target();
        let build = build();
        let translated = cached(
            cache,
            &build,
            expected,
            container,
            victory_shader::emit::translate,
            || target() == expected,
        );
        if target() == expected {
            return translated;
        }
    }
    let error = "host shader capabilities changed during translation".to_owned();
    (Err(error.clone()), CacheStatus::Failed(error))
}

#[cfg(test)]
fn shader_key(build: &[u8; 32], target: [u32; 2], container: &[u8]) -> CacheKey {
    shader_key_for(build, target, b"", container)
}

/// [`shader_key`] with `variant` folded in, for a translation of the same
/// container another way ([`host_shader_thread_limited`]).
fn shader_key_for(build: &[u8; 32], [wave, view_limit]: [u32; 2], variant: &[u8], container: &[u8]) -> CacheKey {
    // Bump the format version if the serialized HostShader or refusal changes.
    CacheKey::from_parts(
        "shader.dxil",
        2,
        &[
            build,
            &wave.to_le_bytes(),
            &view_limit.to_le_bytes(),
            variant,
            container,
        ],
    )
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    target: [u32; 2],
    container: &[u8],
    translate: impl FnOnce(&[u8]) -> Result<HostShader, ShaderError>,
    still_current: impl FnOnce() -> bool,
) -> (Translation, CacheStatus) {
    cached_for(cache, build, target, b"", container, translate, still_current)
}

/// [`cached`] with `variant` folded into the key, for a translation of the
/// same container another way ([`host_shader_thread_limited`]).
#[allow(clippy::too_many_arguments)]
fn cached_for(
    cache: &Cache,
    build: &[u8; 32],
    target: [u32; 2],
    variant: &[u8],
    container: &[u8],
    translate: impl FnOnce(&[u8]) -> Result<HostShader, ShaderError>,
    still_current: impl FnOnce() -> bool,
) -> (Translation, CacheStatus) {
    let key = shader_key_for(build, target, variant, container);
    let path = cache.path_for(&key);
    let read = || {
        std::fs::read(&path)
            .ok()
            .and_then(|bytes| decode_record(&bytes))
    };
    if let Some(entry) = read() {
        return (entry, CacheStatus::Hit);
    }
    let translated = translate(container);
    let persist = matches!(translated, Ok(_) | Err(ShaderError::Unsupported { .. }));
    let result = translated.map_err(|e| e.to_string());
    if !persist || !still_current() {
        return (result, CacheStatus::Unstored);
    }
    let status = match cache.write(&key, &encode_record(&result)) {
        Ok(()) => CacheStatus::Stored,
        Err(_) if read().is_some() => CacheStatus::Hit,
        Err(e) => CacheStatus::Failed(format!("{}: {e:#}", path.display())),
    };
    (result, status)
}

const REFUSED: &[u8; 4] = b"VSHE";

fn encode_record(result: &Translation) -> Vec<u8> {
    let payload = match result {
        Ok(shader) => encode(shader),
        Err(reason) => [REFUSED.as_slice(), reason.as_bytes()].concat(),
    };
    let digest = blake3::hash(&payload);
    let mut record = Vec::with_capacity(digest.as_bytes().len() + payload.len());
    record.extend_from_slice(digest.as_bytes());
    record.extend_from_slice(&payload);
    record
}

fn decode_record(bytes: &[u8]) -> Option<Translation> {
    let (digest, payload) = bytes.split_at_checked(32)?;
    if blake3::hash(payload).as_bytes() != digest {
        return None;
    }
    if let Some(reason) = payload.strip_prefix(REFUSED) {
        return Some(Err(std::str::from_utf8(reason).ok()?.to_owned()));
    }
    decode(payload).map(Ok)
}

const MAGIC: &[u8; 4] = b"VSH1";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists.
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs, &s.root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, s.root_indexed_views.len() as u32);
    for view in &s.root_indexed_views {
        for value in [
            u32::from(view.table_slot),
            u32::from(view.constant_slot),
            view.constant_dword,
            view.byte_offset,
            u32::from(view.shift),
        ] {
            word(&mut out, value);
        }
    }
    out
}

fn decode(bytes: &[u8]) -> Option<HostShader> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let n = r.word()? as usize;
    let mut root_indexed_views = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        root_indexed_views.push(RootIndexedView {
            table_slot: u8::try_from(r.word()?).ok()?,
            constant_slot: u8::try_from(r.word()?).ok()?,
            constant_dword: r.word()?,
            byte_offset: r.word()?,
            shift: u8::try_from(r.word()?).ok().filter(|&s| s < 32)?,
        });
    }
    r.0.is_empty().then_some(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        root_indexed_views,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![1],
            root_indexed_views: vec![RootIndexedView {
                table_slot: 1,
                constant_slot: 2,
                constant_dword: 3,
                byte_offset: 32,
                shift: 5,
            }],
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let target = [32, 999_999];
        let container = b"a console container".as_slice();
        let translate = |_: &[u8]| Ok(shader());
        let entry = || cached(&cache, &build, target, container, translate, || true);
        assert_eq!(entry(), (Ok(shader()), CacheStatus::Stored));
        assert_eq!(entry(), (Ok(shader()), CacheStatus::Hit));
        for other_target in [[64, target[1]], [target[0], 0]] {
            assert_eq!(
                cached(&cache, &build, other_target, container, translate, || true),
                (Ok(shader()), CacheStatus::Stored)
            );
        }
        assert_eq!(
            cached(&cache, &[8; 32], target, container, translate, || true),
            (Ok(shader()), CacheStatus::Stored)
        );

        let path = cache.path_for(&shader_key(&build, target, container));
        let mut damaged = std::fs::read(&path).unwrap();
        damaged[40] ^= 1;
        std::fs::write(&path, damaged).unwrap();
        assert_eq!(entry(), (Ok(shader()), CacheStatus::Stored));
        assert_eq!(entry(), (Ok(shader()), CacheStatus::Hit));
    }

    #[test]
    fn keeps_unsupported_results_but_not_invalid_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let target = [32, 999_999];
        let error = || ShaderError::Unsupported {
            offset: 4,
            what: "test opcode",
        };
        let first = cached(
            &cache,
            &build,
            target,
            b"unsupported",
            |_| Err(error()),
            || true,
        );
        assert_eq!(first, (Err(error().to_string()), CacheStatus::Stored));
        let second = cached(
            &cache,
            &build,
            target,
            b"unsupported",
            |_| panic!("cache miss"),
            || true,
        );
        assert_eq!(second, (Err(error().to_string()), CacheStatus::Hit));
        let invalid = cached(
            &cache,
            &build,
            target,
            b"invalid",
            |_| Err(ShaderError::NotDxbc),
            || true,
        );
        assert_eq!(invalid.1, CacheStatus::Unstored);
        assert!(
            !cache
                .path_for(&shader_key(&build, target, b"invalid"))
                .exists()
        );
    }

    #[test]
    fn cache_write_failure_keeps_the_translation() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let target = [32, 999_999];
        let container = b"shader";
        std::fs::create_dir_all(cache.path_for(&shader_key(&build, target, container))).unwrap();
        let (translated, status) =
            cached(&cache, &build, target, container, |_| Ok(shader()), || true);
        assert_eq!(translated, Ok(shader()));
        assert!(matches!(status, CacheStatus::Failed(_)));
    }

    #[test]
    fn canonical_container_ignores_padding_only_for_valid_header_lengths() {
        let mut bytes = [0; 40];
        bytes[..4].copy_from_slice(b"DXBC");
        bytes[0x18..0x1c].copy_from_slice(&32u32.to_le_bytes());
        assert_eq!(canonical_container(&bytes), &bytes[..32]);
        bytes[0x18..0x1c].copy_from_slice(&41u32.to_le_bytes());
        assert_eq!(canonical_container(&bytes), &bytes);
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader());
        assert_eq!(decode(&bytes), Some(shader()));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
    }
}
