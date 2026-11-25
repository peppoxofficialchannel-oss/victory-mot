//! A lever on the exposure the title applies before its tone curve.
//!
//! The title keeps the scene's exposure at about 1.0 (`PS_CalculateExposure`
//! measures it, `PS_ResetExposure` carries it when nothing measured it), and
//! ACES then puts most of the picture in its knee, where it desaturates: the
//! measured frames lose saturation as they brighten (mean 149 → 0.18 against
//! mean 56 → 0.65). The colour grade that would correct it is not available:
//! the LUTs are cooked blind in the frames the colour pass runs
//! (`_port/REPARTO_COLORE.md`, «Lo slot 624276»). With
//! `VICTORY_EXPOSURE_SCALE=<number>` the exposure the tone pass reads is
//! multiplied by that number, which moves the picture along ACES instead.
//!
//! What it rewrites: the `1x1 f2` image the exposure programs draw into
//! (`PS_CalculateExposure` / `PS_ResetExposure`, a three-vertex draw with a
//! 1x1 viewport), straight after the draw, so every later reader — the tone
//! pass `PS_CompositeNMBNoise` samples it — sees the scaled value. The image
//! carries the exposure twice, `x` and `y` with **`x = 2^y`**: two samples
//! agree to five decimals (`y` -1.3 with `x` 0.40613; `y` -12.9 with `x`
//! 0.00013, where 2^-1.3 = 0.406126). The tone pass reads `y`, so the pass
//! scales `x` by the factor and moves `y` by its log **base 2**, which keeps
//! `x = 2^y` true: at factor 0.5 the read-back trace shows `x` halving and
//! `y` moving by exactly -1.00000. `z` and `w` are left as the title wrote
//! them: `ResetExposure` carries `w` from its input, and `CalculateExposure`
//! puts another value in `z`.
//!
//! The image is a render target and is created without
//! `ALLOW_UNORDERED_ACCESS`, so it has no unordered access view
//! (`descriptor::host::fits` refuses one). The pass therefore copies it into
//! a 1x1 image of its own that shaders can write, scales it there, and copies
//! it back — two 16-byte copies and one thread, once per exposure draw.
//!
//! Off by default (and with `VICTORY_EXPOSURE_SCALE=1`): the title's own
//! exposure stays. Without the factor the pass records nothing at all.
//!
//! `VICTORY_EXPOSURE_LOG=1` reads the 1x1 back as floats, copied just before
//! the scale and just after it, and logs both — the first application and
//! then one every 60. The dumps cannot answer this: they are 8-bit PNGs, and
//! the exposure moves in the decimals of a float whose encoding into that
//! byte we do not know. Decimals are the honest answer. A second line every
//! 60 applications carries `pending`/`logged`, so a trace that is never read
//! cannot go unnoticed again.
//!
//! `VICTORY_EXPOSURE_READERS=1` says which programs name that image in their
//! views, once per program and per side of the pass — the question the ops
//! dumps cannot answer, since they hold only hashes and no reader list. A
//! summary line at the same 60 carries how many distinct programs were seen.
//!
//! The read happens at the *next* application, not when the copying list
//! completes: a `HostList` is moved into `crate::list::Detached` after it is
//! recorded, so the address it has while recording is not the one it has when
//! it runs, and a trace keyed by that address is never found again.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::host::{HostList, host};
use crate::object::This;

/// The programs it follows, by the start of their name (as the title gives
/// the container; the log only has their hashes).
const PROGRAMS: [&[u8]; 2] = [b"PS_CalculateExposure", b"PS_ResetExposure"];

/// The exposure image's format: `f2`, `R32G32B32A32_FLOAT`.
const FORMAT: DXGI_FORMAT = DXGI_FORMAT_R32G32B32A32_FLOAT;

/// The smallest and largest factor it takes: past these the picture is
/// black or blown out whatever the scene, and a mistyped number would be
/// indistinguishable from a bug.
const RANGE: std::ops::RangeInclusive<f32> = 0.05..=4.0;

/// The factor `VICTORY_EXPOSURE_SCALE` asks for; `None` if off.
pub(crate) fn scale() -> Option<f32> {
    static SCALE: OnceLock<Option<f32>> = OnceLock::new();
    *SCALE.get_or_init(|| scale_of(&std::env::var("VICTORY_EXPOSURE_SCALE").unwrap_or_default()))
}

/// The factor `set` asks for: a finite number in [`RANGE`], else none.
///
/// Exactly 1 leaves the exposure as it is (the pass is not recorded).
fn scale_of(set: &str) -> Option<f32> {
    set.trim()
        .parse()
        .ok()
        .filter(|f: &f32| f.is_finite() && RANGE.contains(f))
}

/// Whether the container `bytes` is one of the programs.
pub(crate) fn writes(bytes: &[u8]) -> bool {
    PROGRAMS
        .iter()
        .any(|program| bytes.windows(program.len()).any(|w| w == *program))
}

/// Whether `VICTORY_EXPOSURE_LOG` asks for the 1x1 read back and logged.
pub(crate) fn logging() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_EXPOSURE_LOG").is_some_and(|v| v != "0"))
}

/// Whether `VICTORY_EXPOSURE_READERS` asks which programs name the exposure
/// image in their views.
pub(crate) fn readers() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_EXPOSURE_READERS").is_some_and(|v| v != "0"))
}

/// The exposure image, by its title address, once a program that writes it
/// has drawn it.
static IMAGE: AtomicUsize = AtomicUsize::new(0);
/// The frame the pass last recorded in: `u64::MAX` before any, so no frame
/// is taken for one it scaled.
static SCALED_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
/// The programs seen naming the image: their description, whether they are
/// compute, which side of the pass they read on, and whether they also write
/// it.
static SEEN: Mutex<Vec<(String, bool, bool, bool)>> = Mutex::new(Vec::new());

/// The host state a texture a shader samples needs (pixel and not).
const SHADER_RESOURCE: u32 = 0x40 | 0x80;

/// Whether application `n` (from 1) is one the trace prints: the first, then
/// one every 60.
fn every(n: u64) -> bool {
    n == 1 || n % 60 == 0
}

/// The 1x1 as it was before the scale and after it, kept until the
/// application after it has read them.
struct Trace {
    before: ID3D12Resource,
    after: ID3D12Resource,
    factor: f32,
    n: u64,
    frame: u64,
}

/// Traces its list has not been read from yet, and two counters: how many
/// were registered and how many were read (`pending` is the difference, and
/// stays above zero if a trace is never read).
///
/// The list a trace comes from is not named here: a `HostList` is moved after
/// it is recorded, into `crate::list::Detached { host }`, so the address it
/// had while recording is not the one it has when it runs. The trace is read
/// at the start of the next application instead, whose recording happens
/// after that list was submitted.
static PENDING: Mutex<Vec<Trace>> = Mutex::new(Vec::new());
static REGISTERED: AtomicU64 = AtomicU64::new(0);
static LOGGED: AtomicU64 = AtomicU64::new(0);

/// How many times the scale has been recorded (frames, when the exposure is
/// drawn once a frame).
static APPLICATIONS: AtomicU64 = AtomicU64::new(0);

/// A 512-byte read-back buffer: one 1x1 `f2` row, 256-byte pitched.
fn readback(device: &ID3D12Device) -> windows::core::Result<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 512,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let mut out: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a buffer; a read-back buffer stays a copy
    // destination for its whole life.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut out,
        )?
    };
    out.ok_or_else(windows::core::Error::empty)
}

/// The read-back buffer as the destination of a 1x1 `f2` copy.
fn placed(buffer: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: FORMAT,
                    Width: 1,
                    Height: 1,
                    Depth: 1,
                    RowPitch: 256,
                },
            },
        },
    }
}

/// The four floats of a read-back buffer whose list has run.
fn read(buffer: &ID3D12Resource) -> [f32; 4] {
    let mut data = std::ptr::null_mut();
    let mut out = [0f32; 4];
    // SAFETY: a read-back buffer of ours, written by a list that has run;
    // only the first 16 bytes are read, and the buffer holds 512.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_ok() && !data.is_null() {
            std::ptr::copy_nonoverlapping(data.cast::<f32>(), out.as_mut_ptr(), 4);
            buffer.Unmap(0, None);
        }
    }
    out
}

/// Reads the traces the previous applications left, and logs them. Runs at
/// the start of an application: the list that copied them was submitted a
/// frame earlier, and the buffers are made fresh for each trace, so nothing
/// is being written into them now.
fn read_pending() {
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    for trace in pending.drain(..) {
        let before = read(&trace.before);
        let after = read(&trace.after);
        LOGGED.fetch_add(1, Ordering::Relaxed);
        tracing::info!(
            factor = trace.factor,
            n = trace.n,
            frame = trace.frame,
            "exposure x/y before -> after (VICTORY_EXPOSURE_LOG): {:.5}/{:.5} -> {:.5}/{:.5}",
            before[0],
            before[1],
            after[0],
            after[1]
        );
    }
}

/// The two counters, and how many traces are still waiting to be read.
fn log_counts() {
    let registered = REGISTERED.load(Ordering::Relaxed);
    let logged = LOGGED.load(Ordering::Relaxed);
    let pending = registered.saturating_sub(logged);
    tracing::info!(
        registered,
        "exposure trace pending={} logged={}",
        pending,
        logged
    );
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), DescriptorTable(UAV(u0))"
cbuffer Scale : register(b0) { float factor; float logFactor; float pad0; float pad1; };
RWTexture2D<float4> exposure : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    // The image carries the exposure in x and its log base 2 in y (x = 2^y);
    // the tone pass reads y. Scale x by the factor and move y by its log2,
    // which keeps the two in step (the trace shows x halving and y -1.00000
    // at factor 0.5).
    float4 e = exposure[uint2(0, 0)];
    e.x *= factor;
    e.y += logFactor;
    exposure[uint2(0, 0)] = e;
}
"#;

/// The pass's own image and the objects that write it. One image, one view:
/// unlike the stand-in for indirect light it always writes the same image,
/// so a list in flight can share the view.
struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    views: ID3D12DescriptorHeap,
    /// The 1x1 image of ours the exposure is copied into and back from: a
    /// render target here, unordered access for the pass.
    copy: ID3D12Resource,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn make(device: &ID3D12Device) -> windows::core::Result<Pass> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("exposure"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    let text = |b: &ID3DBlob| {
        // SAFETY: the blob's own bytes.
        unsafe { std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize()) }
            .to_vec()
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            tracing::warn!(
                "the exposure scale pass does not compile: {}",
                String::from_utf8_lossy(&text(&errors))
            );
        }
        return Err(e);
    }
    let code = text(&code.ok_or_else(windows::core::Error::empty)?);
    // SAFETY: the root signature is inside the shader; plain creations.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &code)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: code.as_ptr().cast(),
                BytecodeLength: code.len(),
            },
            ..Default::default()
        })?;
        // The image of ours: 1x1 f2, a render target here and writable by
        // the pass. The title's own image has no unordered access, which is
        // why it is copied through this one.
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
                | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let mut copy: Option<ID3D12Resource> = None;
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut copy,
        )?;
        let copy = copy.ok_or_else(windows::core::Error::empty)?;
        crate::host::name(&copy, || "exposure scale 1x1 f2".to_owned());
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        let views: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: kind,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?;
        let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: FORMAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        device.CreateUnorderedAccessView(
            &copy,
            None::<&ID3D12Resource>,
            Some(&view),
            views.GetCPUDescriptorHandleForHeapStart(),
        );
        Ok(Pass {
            root,
            pso,
            views,
            copy,
        })
    }
}

fn barrier(
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    from: D3D12_RESOURCE_STATES,
    to: D3D12_RESOURCE_STATES,
) {
    // SAFETY: an open list and a live resource.
    unsafe { list.ResourceBarrier(&[crate::host::transition(resource, from, to)]) };
}

/// The image at `at` (subresource 0), for the copies.
fn subresource(texture: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    }
}

/// Keeps the exposure image, by title address, for the reader scan and for
/// the pass: the program that writes it names it in `draw`.
pub(crate) fn note_image(target: This) {
    IMAGE.store(target as usize, Ordering::Relaxed);
}

/// Whether one of a program's views names the exposure image.
pub(crate) fn names_image(view: &crate::descriptor::View) -> bool {
    let image = IMAGE.load(Ordering::Relaxed);
    if image == 0 {
        return false;
    }
    match view {
        crate::descriptor::View::Shader(resource, _) => *resource == image,
        crate::descriptor::View::Unordered(resource, _, _) => *resource == image,
        _ => false,
    }
}

/// Whether the pass has already been recorded in this frame.
pub(crate) fn scaled_this_frame() -> bool {
    SCALED_FRAME.load(Ordering::Relaxed) == crate::frame::current()
}

/// `VICTORY_EXPOSURE_READERS`: `program` names the exposure image in its
/// views. Said once per program, per `compute`, per `writes` and per side of
/// the pass — a program seen both before and after it is printed twice,
/// which is the answer.
pub(crate) fn log_reader(program: &str, compute: bool, scaled: bool, writes: bool) {
    if !readers() || IMAGE.load(Ordering::Relaxed) == 0 {
        return;
    }
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen
        .iter()
        .any(|(seen, seen_compute, seen_scaled, seen_writes)| {
            seen == program
                && *seen_compute == compute
                && *seen_scaled == scaled
                && *seen_writes == writes
        })
    {
        return;
    }
    seen.push((program.to_owned(), compute, scaled, writes));
    tracing::info!(
        program,
        compute,
        scaled,
        writes,
        "exposure reader (VICTORY_EXPOSURE_READERS): a program's views name the exposure image"
    );
}

/// How many programs have been seen naming the exposure image.
fn log_reader_count() {
    let seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let mut programs: Vec<&str> = seen.iter().map(|(program, ..)| program.as_str()).collect();
    programs.sort_unstable();
    programs.dedup();
    tracing::info!(
        readers = programs.len(),
        "exposure readers: {} distinct programs (VICTORY_EXPOSURE_READERS)",
        programs.len()
    );
}

/// The log the image's `y` is moved by: the image keeps `x = 2^y`, so base 2
/// (a natural log would move `y` by less than the factor asks: -0.6931
/// instead of -1 at 0.5, which scales `x` by 0.619, not 0.5).
fn log_of(factor: f32) -> f32 {
    factor.log2()
}

/// Scales by `VICTORY_EXPOSURE_SCALE` the exposure image, in `list`, just
/// before the draw whose views read it. Whether it recorded anything.
///
/// The image is left a shader resource: the draw that asked for the copy is
/// about to sample it. The copy of ours is left in the common state; every
/// run records the same barriers, and lists run in submission order, so a
/// list can assume it common at its start without tracking it.
pub(crate) fn scale_image(list: &HostList) -> bool {
    let (Some(factor), Some(host)) = (scale(), host()) else {
        return false;
    };
    // Exactly 1 is the title's own exposure: not worth a pass.
    if factor == 1.0 {
        return false;
    }
    let image = IMAGE.load(Ordering::Relaxed);
    if image == 0 {
        return false;
    }
    let Some(exposure) = crate::resource::host_resource(image as This) else {
        return false;
    };
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: a live resource.
    let d = unsafe { exposure.GetDesc() };
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || d.Width != 1
        || d.Height != 1
        || d.DepthOrArraySize != 1
        || d.MipLevels != 1
        || d.Format != FORMAT
    {
        // The exposure is the only 1x1 f2 that a draw writes: anything else
        // under an exposure program is not what this lever knows how to
        // scale, and scaling it would be a guess.
        ONCE.call_once(|| {
            tracing::warn!(
                format = d.Format.0,
                width = d.Width,
                height = d.Height,
                "no exposure scale (VICTORY_EXPOSURE_SCALE): its image is not the 1x1 f2 exposure"
            )
        });
        return false;
    }
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    let Some(pass) = PASS
        .get_or_init(|| {
            make(&host.device)
                .inspect_err(
                    |e| tracing::warn!(error = %e, "no exposure scale (VICTORY_EXPOSURE_SCALE)"),
                )
                .ok()
        })
        .as_ref()
    else {
        return false;
    };
    ONCE.call_once(|| {
        tracing::info!(
            factor,
            "the exposure is scaled before the tone curve (VICTORY_EXPOSURE_SCALE)"
        )
    });
    let l = &list.list;
    list.keep(&pass.copy);
    // `VICTORY_EXPOSURE_LOG`: first what the previous application left (its
    // list has run by now), then this application's own trace, copied before
    // the scale and after it.
    let mut trace: Option<(u64, ID3D12Resource, ID3D12Resource)> = None;
    if logging() || readers() {
        if logging() {
            read_pending();
        }
        let n = APPLICATIONS.fetch_add(1, Ordering::Relaxed) + 1;
        if every(n) {
            if logging() {
                log_counts();
            }
            if readers() {
                log_reader_count();
            }
            if logging() {
                match (readback(&host.device), readback(&host.device)) {
                    (Ok(before), Ok(after)) => {
                        list.keep(&before);
                        list.keep(&after);
                        trace = Some((n, before, after));
                    }
                    // Without the buffers there is nothing to read: said
                    // once, since `registered` would then stay at zero.
                    _ => {
                        static ONCE: std::sync::Once = std::sync::Once::new();
                        ONCE.call_once(|| {
                            tracing::warn!(
                                "no exposure trace (VICTORY_EXPOSURE_LOG): its read-back buffers are not made"
                            )
                        });
                    }
                }
            }
        }
    }
    // The image, put where the copy reads it (the reader that asked for the
    // pass has already had it a shader resource, `readable`).
    crate::list::ensure_state(
        list,
        image as This,
        D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32,
    );
    // SAFETY: an open list and live resources; our image is common at the
    // start of every recording and left so at its end.
    unsafe {
        barrier(
            l,
            &pass.copy,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        l.CopyTextureRegion(
            &subresource(&pass.copy),
            0,
            0,
            0,
            &subresource(exposure),
            None,
        );
        match &trace {
            // The value the title wrote, before the scale.
            Some((_, before, _)) => {
                barrier(
                    l,
                    &pass.copy,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                );
                l.CopyTextureRegion(&placed(before), 0, 0, 0, &subresource(&pass.copy), None);
                barrier(
                    l,
                    &pass.copy,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
            }
            None => barrier(
                l,
                &pass.copy,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
        }
        l.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        l.SetComputeRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        // Base 2: the image keeps `x = 2^y`, and the tone pass reads `y`.
        let constants = [factor, log_of(factor), 0.0f32, 0.0];
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(1, pass.views.GetGPUDescriptorHandleForHeapStart());
        l.Dispatch(1, 1, 1);
        // The barrier into a copy state also orders the writes the pass made.
        barrier(
            l,
            &pass.copy,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        // The value as the pass left it, after the scale.
        if let Some((_, _, after)) = &trace {
            l.CopyTextureRegion(&placed(after), 0, 0, 0, &subresource(&pass.copy), None);
        }
    }
    crate::list::ensure_state(list, image as This, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32);
    // SAFETY: as above; the scaled image goes back into it.
    unsafe {
        l.CopyTextureRegion(
            &subresource(exposure),
            0,
            0,
            0,
            &subresource(&pass.copy),
            None,
        );
    }
    // A shader resource, for the draw this pass runs before.
    crate::list::ensure_state(list, image as This, SHADER_RESOURCE);
    barrier(
        l,
        &pass.copy,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COMMON,
    );
    SCALED_FRAME.store(crate::frame::current(), Ordering::Relaxed);
    if let Some((n, before, after)) = trace {
        PENDING
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Trace {
                before,
                after,
                factor,
                n,
                frame: crate::frame::current(),
            });
        REGISTERED.fetch_add(1, Ordering::Relaxed);
    }
    true
}

#[cfg(test)]
mod tests {
    /// Off unless a factor in the range is asked for.
    #[test]
    fn the_factor_is_asked_for() {
        assert_eq!(super::scale_of(""), None);
        assert_eq!(super::scale_of("off"), None);
        assert_eq!(super::scale_of("0"), None);
        assert_eq!(super::scale_of("-1"), None);
        assert_eq!(super::scale_of("nan"), None);
        assert_eq!(super::scale_of("inf"), None);
        // Outside 0.05..4.0: refused, not clamped.
        assert_eq!(super::scale_of("0.04"), None);
        assert_eq!(super::scale_of("4.01"), None);
        assert_eq!(super::scale_of("100"), None);
        // The ends of the range and inside it.
        assert_eq!(super::scale_of("0.05"), Some(0.05));
        assert_eq!(super::scale_of("4.0"), Some(4.0));
        assert_eq!(super::scale_of(" 0.5 "), Some(0.5));
        assert_eq!(super::scale_of("1.0"), Some(1.0));
    }

    /// The trace prints the first application, then one every 60.
    #[test]
    fn the_trace_prints_one_in_sixty() {
        assert!(super::every(1));
        assert!(!super::every(2));
        assert!(!super::every(59));
        assert!(super::every(60));
        assert!(!super::every(61));
        assert!(super::every(120));
        assert!(super::every(121 + 59));
    }

    /// The programs are found by name, anywhere in the container.
    #[test]
    fn the_exposure_programs_are_found_by_name() {
        assert!(super::writes(
            b"....PS_CalculateExposure_396dfccc_Wrapped\0"
        ));
        assert!(super::writes(b"....PS_ResetExposure_396dfccc_Wrapped\0"));
        assert!(!super::writes(
            b"....PS_CompositeNMBNoise_97148495_Wrapped\0"
        ));
        assert!(!super::writes(b""));
    }

    /// The log moved is base 2: the image keeps `x = 2^y`, and at 0.5 the
    /// move is exactly -1 (a natural log would give -0.6931, and `x` would
    /// land at 0.619 of its value, not 0.5).
    #[test]
    fn the_log_of_the_factor_is_base_two() {
        assert_eq!(super::log_of(0.5), -1.0);
        assert_eq!(super::log_of(0.25), -2.0);
        assert_eq!(super::log_of(4.0), 2.0);
        assert_eq!(super::log_of(1.0), 0.0);
    }

    /// The pass compiles with the system's compiler and the host takes it.
    #[test]
    fn exposure_pass_builds() {
        let Some(host) = crate::host::host() else {
            return;
        };
        super::make(&host.device).expect("the exposure pass");
    }
}
