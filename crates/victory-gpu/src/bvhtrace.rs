//! Diagnostics: a trace of what the ray-tracing structure builder and the
//! other programs that reach title memory by address do
//! (`VICTORY_BVH_TRACE=<file>`, diagnostics builds only). One line per
//! event, each led by the frame number:
//!
//! - `D`: a dispatch of such a program (its shader hash, group counts, root
//!   constants, root views with the first 64 bytes of title memory there);
//! - `N`: a copy given a number in the directory of title memory by
//!   address (`crate::memory`), with why;
//! - `F`: the pages shaders touched that had no copy;
//! - `W`: from the builder's first dispatch on, every range of title
//!   memory a submission's shader writes were put back into, with its
//!   first bytes;
//! - `V`: what walking the structures the builder made on the card found,
//!   `V rebuilt` for those it rebuilt in place from what was streamed
//!   (`VICTORY_BVH_CHECK`, [`check_stages`]), once their header was put
//!   back (`late`: walked with no put-back seen, [`check_due`]).
//!
//! The file is written unbuffered: a run that ends in the title's crash
//! keeps every line written before it.

use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Lines written at most: a trace of the first frames of the builder (a
/// frame of it is some 2,000 dispatches and thousands of put-back ranges).
const LIMIT: u64 = 2_000_000;

static FILE: Mutex<Option<std::fs::File>> = Mutex::new(None);
static LINES: AtomicU64 = AtomicU64::new(0);
static ARMED: AtomicBool = AtomicBool::new(false);

fn path() -> Option<&'static std::path::PathBuf> {
    static PATH: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| std::env::var_os("VICTORY_BVH_TRACE").map(std::path::PathBuf::from))
        .as_ref()
}

/// Whether the trace is asked for.
pub(crate) fn on() -> bool {
    path().is_some()
}

/// The builder has dispatched: put-back ranges are traced from now on.
pub(crate) fn arm() {
    if on() {
        ARMED.store(true, Ordering::Relaxed);
    }
}

/// Whether put-back ranges are traced.
pub(crate) fn armed() -> bool {
    ARMED.load(Ordering::Relaxed)
}

/// Adds a line.
pub(crate) fn line(what: impl FnOnce() -> String) {
    let Some(path) = path() else {
        return;
    };
    if LINES.fetch_add(1, Ordering::Relaxed) >= LIMIT {
        return;
    }
    check_due();
    let text = format!("{} {}\n", crate::frame::current(), what());
    let mut file = FILE.lock().unwrap_or_else(|e| e.into_inner());
    if file.is_none() {
        *file = std::fs::File::create(path).ok();
    }
    if let Some(f) = file.as_mut() {
        let _ = f.write_all(text.as_bytes());
    }
}

/// Up to `n` bytes of title memory at `address`, as hex dwords ("-" if not
/// readable).
pub(crate) fn peek(address: u64, n: usize) -> String {
    let mut bytes = vec![0u8; n];
    if !crate::host::read_into(address as *const u8, &mut bytes) {
        return "-".to_owned();
    }
    hex(&bytes)
}

/// `bytes` as little-endian hex dwords (the rest as bytes).
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut text = String::with_capacity(bytes.len() * 9 / 4 + 4);
    let mut words = bytes.chunks_exact(4);
    for w in &mut words {
        let _ = write!(
            text,
            "{:08x} ",
            u32::from_le_bytes([w[0], w[1], w[2], w[3]])
        );
    }
    for b in words.remainder() {
        let _ = write!(text, "{b:02x}");
    }
    text.trim_end().to_owned()
}

/// `VICTORY_BVH_CHECK=<hash>[,<hash>]` (or `+` between them, for scripts
/// that split their switches at commas): the builder stages whose structures
/// are walked a frame later ([`check_due`]), a `V` line saying what was
/// found. The first writes structure headers (InitHeaders: its first root
/// view is an array of 40-byte records in list upload memory, each naming a
/// header at +0, and the first dword of its root constants their count):
/// the structures built on the card. The second, if given, starts the
/// rebuilding of a pre-built structure in place, its header at its root
/// view 1 (DecompressHeader): the structures streamed from disk. Given at
/// run time: the stages' hashes are the title's.
fn check_stages() -> &'static [String] {
    static HASHES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    HASHES.get_or_init(|| {
        std::env::var("VICTORY_BVH_CHECK")
            .ok()
            .filter(|_| on())
            .map(|v| v.split([',', '+']).map(|h| h.trim().to_owned()).filter(|h| !h.is_empty()).collect())
            .unwrap_or_default()
    })
}

/// The stage writing the headers of structures built on the card
/// ([`check_stages`]).
pub(crate) fn header_stage() -> Option<&'static str> {
    check_stages().first().map(String::as_str)
}

/// The stage starting the rebuilding of a pre-built structure in place
/// ([`check_stages`]).
pub(crate) fn rebuild_stage() -> Option<&'static str> {
    check_stages().get(1).map(String::as_str)
}

/// A header named: the frame it was named in, its address, whether the
/// structure is rebuilt in place, and the frame the first put-back over the
/// header after that landed in (0: none yet; [`put_back`]).
type Named = (u64, u64, bool, u64);

/// Headers named and not walked yet.
static NAMED: Mutex<Vec<Named>> = Mutex::new(Vec::new());

/// Frames after which a structure is walked even though no put-back over
/// its header was seen (counted as `late` in its `V` line).
const LATE: u64 = 16;

/// Notes the structure a rebuilding stage starts on, whose header is at
/// `header` ([`rebuild_stage`]).
pub(crate) fn rebuilt_named(header: u64) {
    let frame = crate::frame::current();
    let mut named = NAMED.lock().unwrap_or_else(|e| e.into_inner());
    if header != 0 && named.len() < 1 << 16 {
        named.push((frame, header, true, 0));
    }
}

/// Title memory `at`..`at + len` was put back from what shaders wrote:
/// structures named before, whose header lies there, can be walked from the
/// next frame on. The walk reads title memory, which holds the builder's
/// writes only once they are put back after their submission ends, a frame
/// or more after the dispatches were traced (run integ-general-2-rt2-builder:
/// a structure rebuilt in frame 3255 was walked at the start of 3256 still
/// in its streamed form, and put back a few lines later).
pub(crate) fn put_back(at: u64, len: u64) {
    if check_stages().is_empty() {
        return;
    }
    let frame = crate::frame::current();
    let mut named = NAMED.lock().unwrap_or_else(|e| e.into_inner());
    for n in named.iter_mut() {
        if n.3 == 0 && n.0 <= frame && (at..at.saturating_add(len)).contains(&n.1) {
            n.3 = frame;
        }
    }
}
/// The frame [`check_due`] last ran in.
static CHECKED: AtomicU64 = AtomicU64::new(0);

/// Notes the headers a header stage's `count` records at `records` name
/// (read at submission, when the builder has filled them).
pub(crate) fn headers_named(records: u64, count: u32) {
    let count = count.min(1 << 16) as usize;
    let mut bytes = vec![0u8; count * 40];
    if count == 0 || !crate::host::read_into(records as *const u8, &mut bytes) {
        return;
    }
    let frame = crate::frame::current();
    let mut named = NAMED.lock().unwrap_or_else(|e| e.into_inner());
    for record in bytes.chunks_exact(40) {
        let header = u64::from_le_bytes(record[0..8].try_into().unwrap_or_default());
        if header != 0 && named.len() < 1 << 16 {
            named.push((frame, header, false, 0));
        }
    }
}

/// What walking structures found.
#[derive(Default)]
struct Walk {
    structures: u64,
    empty: u64,
    unreadable: u64,
    boxes: u64,
    triangles: u64,
    instances: u64,
    bad: u64,
    /// Structures walked with no put-back over their header seen ([`LATE`]).
    late: u64,
    /// The first problems, with where.
    first: Vec<String>,
}

impl Walk {
    fn bad(&mut self, header: u64, at: u64, what: &str) {
        self.bad += 1;
        if self.first.len() < 12 {
            self.first.push(format!("{header:#x}@{at:#x}: {what}"));
        }
    }
}

/// Whether box `b` (minimum x y z, maximum x y z) lies in `outer`, with
/// room for 16-bit rounding (boxes of 16-bit nodes are rounded outwards).
fn inside(b: &[f32; 6], outer: &[f32; 6]) -> bool {
    let slack = |v: f32| v.abs().max(1.0) * (1.0 / 256.0);
    (0..3).all(|k| {
        b[k] >= outer[k] - slack(outer[k]) && b[k + 3] <= outer[k + 3] + slack(outer[k + 3])
    })
}

/// Walks the structure whose header is at `header`: every box finite, not
/// inverted and inside its parent's box (the root's inside the header's
/// bounds), every triangle's vertices finite and inside their box, every
/// node inside the structure's size and reached once.
///
/// The nodes of a structure `rebuilt` in place are not in the header's
/// space: the leaf rehydration writes each vertex as (M v - o) s, with o
/// and s from its root constants (dwords 0-2 and 4-6, s powers of two such
/// as 2^18, 2^18, 2^17), and the boxes above are made from those vertices,
/// while the header keeps the bounds in the structure's own space. Their
/// root is not held to the header's bounds (in run
/// integ-general-2-rt2-builder every root child "outside" them was exactly
/// that scale, axis by axis).
fn walk(header: u64, rebuilt: bool, w: &mut Walk) {
    let mut h = [0u8; 64];
    if !crate::host::read_into(header as *const u8, &mut h) {
        w.unreadable += 1;
        return;
    }
    let word =
        |b: &[u8], k: usize| u32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap_or_default());
    w.structures += 1;
    let root = word(&h, 12);
    if root == u32::MAX {
        w.empty += 1;
        return;
    }
    let size = (u64::from(word(&h, 1)) + 1) * 64;
    let bounds: [f32; 6] = std::array::from_fn(|k| f32::from_bits(word(&h, 4 + k)));
    if !bounds.iter().all(|v| v.is_finite()) || (0..3).any(|k| bounds[k] > bounds[k + 3]) {
        w.bad(header, header, &format!("header bounds {bounds:?}"));
    }
    let mut seen = std::collections::HashSet::new();
    let everywhere = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY, f32::INFINITY];
    let mut stack = vec![(root, if rebuilt { everywhere } else { bounds })];
    let mut nodes = 0u32;
    while let Some((pointer, outer)) = stack.pop() {
        nodes += 1;
        if nodes > 1 << 20 {
            w.bad(header, header, "more than a million nodes");
            return;
        }
        let kind = pointer & 7;
        let at = header + (u64::from(pointer >> 3) << 6);
        if !seen.insert(pointer) {
            w.bad(header, at, "a node reached twice");
            continue;
        }
        let len: u64 = if kind == 5 { 128 } else { 64 };
        if at + len > header + size {
            w.bad(
                header,
                at,
                &format!("a node past the structure's {size:#x} bytes (pointer {pointer:#x})"),
            );
            continue;
        }
        // The node's first 28 dwords: all an instruction reads.
        let mut bytes = [0u8; 112];
        if !crate::host::read_into(at as *const u8, &mut bytes[..len.min(112) as usize]) {
            w.bad(header, at, "a node not readable");
            continue;
        }
        let words: [u32; 28] = std::array::from_fn(|k| word(&bytes, k));
        match kind {
            4 | 5 => {
                w.boxes += 1;
                let boxes = if kind == 5 {
                    victory_shader::bvh::boxes32(&words)
                } else {
                    victory_shader::bvh::boxes16(&words)
                };
                for (child, b) in victory_shader::bvh::children(&words).into_iter().zip(boxes) {
                    if child == u32::MAX {
                        continue;
                    }
                    if !b.iter().all(|v| v.is_finite()) || (0..3).any(|k| b[k] > b[k + 3]) {
                        w.bad(header, at, &format!("child {child:#x} box {b:?}"));
                    } else if !inside(&b, &outer) {
                        w.bad(
                            header,
                            at,
                            &format!("child {child:#x} box {b:?} outside {outer:?}"),
                        );
                    }
                    stack.push((child, b));
                }
            }
            0..=3 => {
                w.triangles += 1;
                for v in victory_shader::bvh::triangle_vertices(&words, kind) {
                    let point = [v[0], v[1], v[2], v[0], v[1], v[2]];
                    if !v.iter().all(|c| c.is_finite()) {
                        w.bad(header, at, &format!("triangle {kind} vertex {v:?}"));
                    } else if !inside(&point, &outer) {
                        w.bad(
                            header,
                            at,
                            &format!("triangle {kind} vertex {v:?} outside {outer:?}"),
                        );
                    }
                }
            }
            6 => w.instances += 1,
            _ => {}
        }
    }
}

/// Walks, once a frame, the structures whose header was put back in an
/// earlier frame after they were named ([`put_back`]; or named [`LATE`]
/// frames ago with no put-back seen), and says what it found in a `V` line
/// for those built on the card and a `V rebuilt` line for those rebuilt in
/// place. A structure the title rebuilds every frame may still change
/// under the walk: the next frame's put-back runs on another thread.
fn check_due() {
    if check_stages().is_empty() {
        return;
    }
    let frame = crate::frame::current();
    if CHECKED.swap(frame, Ordering::Relaxed) == frame {
        return;
    }
    // (header, rebuilt, late)
    let due: Vec<(u64, bool, bool)> = {
        let mut named = NAMED.lock().unwrap_or_else(|e| e.into_inner());
        let (due, later): (Vec<_>, Vec<_>) = named
            .drain(..)
            .partition(|&(at, _, _, landed)| (landed != 0 && landed < frame) || at + LATE < frame);
        *named = later;
        let mut due: Vec<(u64, bool, bool)> =
            due.into_iter().map(|(_, h, rebuilt, landed)| (h, rebuilt, landed == 0)).collect();
        due.sort_unstable();
        due.dedup_by_key(|d| (d.0, d.1));
        due
    };
    for rebuilt in [false, true] {
        let headers: Vec<(u64, bool)> = due.iter().filter(|d| d.1 == rebuilt).map(|d| (d.0, d.2)).collect();
        if !headers.is_empty() {
            walk_line(&headers, rebuilt);
        }
    }
}

/// Walks the structures at `headers` (each with whether it is walked
/// late) and says what it found in a `V` line (`V rebuilt` for structures
/// `rebuilt` in place).
fn walk_line(headers: &[(u64, bool)], rebuilt: bool) {
    let mut w = Walk::default();
    for &(header, late) in headers {
        w.late += u64::from(late);
        walk(header, rebuilt, &mut w);
    }
    let kind = if rebuilt { "V rebuilt" } else { "V" };
    line(|| {
        format!(
            "{kind} structures {} empty {} unreadable {} box nodes {} triangles {} instances {} bad {} late {}{}",
            w.structures,
            w.empty,
            w.unreadable,
            w.boxes,
            w.triangles,
            w.instances,
            w.bad,
            w.late,
            w.first
                .iter()
                .map(|f| format!(" | {f}"))
                .collect::<String>()
        )
    });
}
