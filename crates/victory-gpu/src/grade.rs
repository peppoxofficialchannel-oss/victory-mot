//! A pass of our own over the finished colour, starting from the identity.
//!
//! `VICTORY_GRADE=identity` records a full-screen triangle that reads the
//! image the title's 8-bit encode samples and writes the **same values**
//! back, just before that draw. No curve, no saturation, no parameter: its
//! only job is the neutral proof, that the hook itself changes nothing. A
//! picture whose p25, mean, saturation, clipping and levels are identical
//! with the switch on and off says the place is right; any difference the
//! curve shows afterwards is then the curve's.
//!
//! `VICTORY_GRADE=curve` records the same triangle with **one** tone curve
//! and a croma, per RGB channel, alpha untouched. Below `VICTORY_GRADE_KNEE`
//! (0.75 by default) a channel is written as it is; above it the range left
//! between the knee and `VICTORY_GRADE_WHITE` (1.0 by default) is bent,
//! continuously and monotonically, so the knee does not move and 1.0 writes
//! the white ([`CURVE_SOURCE`] has the formula). After the curve the croma
//! is pulled toward the channel's luminance by `VICTORY_GRADE_SAT` (1.0 by
//! default, the identity). The four knobs are read once ([`curve`]) and
//! handed to the pixel program as **root constants**, so they change without
//! a rebuild; the identity records no constant at all and keeps its own
//! pipeline.
//!
//! Where it goes, from the dumps (`_port/victory/seriesS_up`):
//!
//! * the finished colour is **not one image**. The composite
//!   `5970ee6dbc47ec61` writes `0x1f68fe5aa30` (1920x1080 `f24`), but in
//!   frames 3509, 3533 and 3583 `43b4c00d725ac9c7` **reads** it and writes
//!   `0x1f68fe5cab0`, and `5510a9fd27ceb4b1` writes that one again. The
//!   colour therefore ping-pongs, and the 8-bit encode `b30b0ce52e51bc0c`
//!   samples `0x1f68fe5aa30` in frame 3505 and `0x1f68fe5cab0` in the other
//!   three (its own table `s1+0`).
//! * `43b4c00d725ac9c7` has `0x1f68fe5aa30` among the views it **reads**,
//!   never among its targets (its target is `0x1f68fe5cab0`): hooking on it
//!   as a writer would wait for a write that never happens.
//! * the last writer of `0x1f68fe5aa30` is the composite in frames 3505,
//!   3509 and 3533, and `edc1cad8ffb79bc9` in 3583 (which draws it once
//!   before the composite and once after).
//!
//! So the one place that is after **every** writer and before the first
//! reader of the whole image is the encode itself: the pass is recorded
//! just before that draw (a draw of `b30b0ce52e51bc0c`, the pixel program
//! [`ENCODE`]), and it takes the image from the draw's own table
//! (`s1+0`, [`COLOUR`]). Whatever image the title leaves the last colour
//! in, the encode names it, and the pass follows it.
//!
//! How: the image cannot be read and written in the same draw (the read
//! would be undefined), and D3D12 copies neither into nor from a resource
//! bound as a target (the exposure pass found this first, commit 59d073a1).
//! So the image is first **copied into** our own image (`pass.copy`, same
//! size and format, `COPY_SOURCE` → `COPY_DEST`) and the draw reads that
//! copy (`SRV`) and writes the title's image itself (`RTV`). The copy is
//! therefore always filled before anything reads it, and nothing is ever
//! copied out of a pass's own target. The image is let go as a target
//! before the copy runs and bound again as one for the draw; `crate::draw`'s
//! `replay` sets the title's targets, viewports and scissors again as soon
//! as this module returns.
//!
//! Off by default, and with `VICTORY_GRADE` set to anything else: [`on`]
//! is false, nothing is recorded and no state is touched.

use std::sync::OnceLock;

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::host::{HostList, host};
use crate::object::This;

/// The program the pass follows, by its hash as the listings give it: the
/// title's 8-bit encode, the last reader of the finished colour.
pub(crate) const ENCODE: &str = "b30b0ce52e51bc0c";

/// Where that program reads the colour: root parameter (a table) and place
/// in its table, in dwords. From the listings: `b30b0ce52e51bc0c Srv s1+0`.
pub(crate) const COLOUR: (u8, u32) = (1, 0);

/// The one value `VICTORY_GRADE` takes for the pass that leaves the
/// picture as it is.
const IDENTITY: &str = "identity";

/// The value `VICTORY_GRADE` takes for the highlight curve.
const CURVE: &str = "curve";

/// Which pass `VICTORY_GRADE` asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Style {
    /// The same values read and written back: no constant, the pipeline of
    /// [`SOURCE`].
    Identity,
    /// The soft-knee curve of [`CURVE_SOURCE`], fed by [`curve`].
    Curve,
}

impl Style {
    /// The word `VICTORY_GRADE` holds for this style, as the logs give it.
    fn name(self) -> &'static str {
        match self {
            Style::Identity => IDENTITY,
            Style::Curve => CURVE,
        }
    }
}

/// The pass `VICTORY_GRADE` asks for (`identity` or `curve`, any case,
/// spaces trimmed). Absent or anything else: `None`, nothing is recorded.
pub(crate) fn style() -> Option<Style> {
    static STYLE: OnceLock<Option<Style>> = OnceLock::new();
    *STYLE.get_or_init(|| {
        std::env::var("VICTORY_GRADE")
            .ok()
            .as_deref()
            .and_then(style_of)
    })
}

/// Whether `set` (what the variable holds) asks for the identity.
fn is_identity(set: &str) -> bool {
    set.trim().eq_ignore_ascii_case(IDENTITY)
}

/// Whether `set` asks for the curve.
fn is_curve(set: &str) -> bool {
    set.trim().eq_ignore_ascii_case(CURVE)
}

/// What `set` asks for; anything it does not know is `None`.
fn style_of(set: &str) -> Option<Style> {
    if is_identity(set) {
        Some(Style::Identity)
    } else if is_curve(set) {
        Some(Style::Curve)
    } else {
        None
    }
}

/// Whether any pass is asked for (`VICTORY_GRADE=identity|curve`, any case,
/// spaces trimmed). Absent or anything else: off.
pub(crate) fn on() -> bool {
    style().is_some()
}

/// The knee `VICTORY_GRADE_KNEE` defaults to: below it a channel is written
/// as it is, which leaves the low three quarters of the picture alone.
const KNEE: f32 = 0.75;

/// The knee is allowed here, both ends included. Above 0.95 the shoulder
/// would be too thin to bend; below 0.3 it would reach into the midtones.
const KNEE_RANGE: (f32, f32) = (0.3, 0.95);

/// How hard the curve bends above the knee (`VICTORY_GRADE_STRENGTH`).
/// 0 is the identity, 2 the default.
const STRENGTH: f32 = 2.0;

/// The strength is allowed here, both ends included: the curve stays
/// monotone over the whole range.
const STRENGTH_RANGE: (f32, f32) = (0.0, 20.0);

/// The value the white ends in (`VICTORY_GRADE_WHITE`): 1.0 writes the top
/// of the range as white, which is the curve as it was before the knob.
const WHITE: f32 = 1.0;

/// The white is allowed here, both ends included. It is not a range on its
/// own: it must stay **above the knee** ([`shoulder`]), or the shoulder would
/// have nowhere to bend and the curve would no longer be monotone.
const WHITE_RANGE: (f32, f32) = (0.5, 1.0);

/// The croma `VICTORY_GRADE_SAT` defaults to: 1.0 is the identity, exactly.
const SAT: f32 = 1.0;

/// The saturation is allowed here, both ends included: 0.0 is grey.
const SAT_RANGE: (f32, f32) = (0.0, 1.0);

/// The four numbers the curve reads, as the pixel program sees them: `b0`,
/// four dwords = 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Knobs {
    knee: f32,
    strength: f32,
    white: f32,
    sat: f32,
}

/// The curve's knobs, read **once** from the environment ([`knob`]) and
/// then carried to the shader as root constants.
fn curve() -> Knobs {
    static KNOBS: OnceLock<Knobs> = OnceLock::new();
    *KNOBS.get_or_init(|| {
        let knee = knob("VICTORY_GRADE_KNEE", KNEE, KNEE_RANGE);
        Knobs {
            knee,
            strength: knob("VICTORY_GRADE_STRENGTH", STRENGTH, STRENGTH_RANGE),
            white: shoulder(knee, knob("VICTORY_GRADE_WHITE", WHITE, WHITE_RANGE)),
            sat: knob("VICTORY_GRADE_SAT", SAT, SAT_RANGE),
        }
    })
}

/// The white the curve is fed: `white` when it sits above `knee`, otherwise
/// 1.0 with a warning. It is **not** a silent clamp: with `white <= knee` the
/// shoulder `knee + (white - knee) g(t)` collapses or inverts — the curve is
/// no longer monotone — so the whole knob falls back to the neutral 1.0 and
/// the log says so.
fn shoulder(knee: f32, white: f32) -> f32 {
    if white <= knee {
        tracing::warn!(
            knee,
            white,
            "the grade's white is not above its knee, so the curve would not be monotone: white = 1.0 is used"
        );
        1.0
    } else {
        white
    }
}

/// One knob: the variable's value when it parses into `range` (inclusive),
/// otherwise `default` — a number outside the range or a word is not a
/// value, and the log says the default was used.
fn knob(name: &str, default: f32, range: (f32, f32)) -> f32 {
    let Ok(set) = std::env::var(name) else {
        return default;
    };
    let Some(value) = knob_of(&set, range) else {
        tracing::warn!(
            variable = name,
            set,
            default,
            low = range.0,
            high = range.1,
            "no such value for the grade curve: the default is used"
        );
        return default;
    };
    value
}

/// `set` as a knob: a finite number inside `range` (inclusive), or `None`.
/// The empty variable is `None` too, not zero.
fn knob_of(set: &str, (low, high): (f32, f32)) -> Option<f32> {
    let value: f32 = set.trim().parse().ok()?;
    (value.is_finite() && (low..=high).contains(&value)).then_some(value)
}

const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0))"
Texture2D<float4> colour : register(t0);
struct Out { float4 at : SV_Position; };
// A triangle larger than the screen: three vertices, no buffer.
[RootSignature(RS)]
Out vs(uint id : SV_VertexID) {
    float2 p = float2(float((id << 1) & 2), float(id & 2));
    Out o;
    o.at = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}
[RootSignature(RS)]
float4 ps(Out i) : SV_Target {
    // The identity: the texel this pixel covers, as it is (the render
    // target's own format carries it back exactly: a unorm read is exact,
    // and the write rounds to the same code).
    return colour.Load(int3(int2(i.at.xy), 0));
}
"#;

/// The curve's program: the same triangle, one tone curve and a croma,
/// alpha untouched.
///
/// Per channel, with `t = (c - knee) / (1 - knee)` and `s = strength`:
/// `g(t) = t + s t^2 (1 - t) / (1 + s t)`, and the channel is written as
/// `knee + (white - knee) g(t)` above the knee, as it is below it. Then
/// `g(0) = 0` (the knee does not move), `g(1) = 1` (1.0 writes `white`) and
/// `g'(0) = 1`; `s t^2 (1 - t)` is never negative, so for `s >= 0` the
/// shoulder bends toward the white and never away, and `s = 0` is linear.
/// With `white = 1.0` this is exactly the curve of before.
///
/// **After** the tone curve, the croma: `l = dot(rgb, luma)` and
/// `rgb = lerp(rgb, l.xxx, 1 - sat)`. `lerp(a, b, 0.0)` returns `a`
/// **exactly**, so `sat = 1.0` is the exact identity, not "almost"; and the
/// luminance `l` is a convex combination of the three channels (the weights
/// sum to one and are positive), so pulling each channel toward `l` can
/// never overshoot the min/max of the three — the desaturation cannot create
/// a new clip.
const CURVE_SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), DescriptorTable(SRV(t0))"
cbuffer Knobs : register(b0) { float knee; float strength; float white; float sat; };
Texture2D<float4> colour : register(t0);
struct Out { float4 at : SV_Position; };
// A triangle larger than the screen: three vertices, no buffer.
[RootSignature(RS)]
Out vs(uint id : SV_VertexID) {
    float2 p = float2(float((id << 1) & 2), float(id & 2));
    Out o;
    o.at = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}
[RootSignature(RS)]
float4 ps(Out i) : SV_Target {
    float4 c = colour.Load(int3(int2(i.at.xy), 0));
    // Where the channel sits above the knee: 0 at the knee, 1 at 1.0.
    float3 t = saturate((c.rgb - knee) / (1.0 - knee));
    // g(t): 0 at the knee, 1 at 1.0, slope 1 at the knee, and for
    // strength >= 0 bent toward white.
    float3 g = t + strength * t * t * (1.0 - t) / (1.0 + strength * t);
    // The shoulder ends at `white` (and at `knee` it is still `knee`).
    float3 bent = knee + (white - knee) * g;
    // Below the knee t is 0 and the texel is written as it is; alpha always.
    c.rgb = lerp(c.rgb, bent, step(knee, c.rgb));
    // Then the croma: toward the luminance, by (1 - sat). sat = 1.0 is the
    // exact identity; l is a convex combination, so this cannot clip anew.
    float l = dot(c.rgb, float3(0.2126, 0.7152, 0.0722));
    c.rgb = lerp(c.rgb, l.xxx, 1.0 - sat);
    return c;
}
"#;

struct Pass {
    /// Which pass this is: it picks the shader, the root signature and the
    /// constants the recording sets.
    style: Style,
    /// The curve's knobs; `Default` (and never set) for the identity.
    knobs: Knobs,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// The shader-visible heap holding the one view the pass reads.
    views: ID3D12DescriptorHeap,
    /// Our copy of the colour: the copy is filled from the image, then read
    /// by the draw. Kept in the common state between runs.
    copy: ID3D12Resource,
    width: u32,
    height: u32,
    /// The image's own format, which our copy is made in: both sides of the
    /// copy name one format.
    format: DXGI_FORMAT,
    /// The format the views name: `format`, or the encode's concrete format
    /// when `format` is typeless (a view cannot name a typeless one).
    view: DXGI_FORMAT,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

/// One shader entry point of `source`, compiled.
fn compile(
    source: &str,
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> windows::core::Result<Vec<u8>> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let made = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("grade"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = made {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!(
                "the grade pass does not compile: {}",
                String::from_utf8_lossy(text)
            );
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes.
    Ok(unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    }
    .to_vec())
}

/// Whether `format` is one of DXGI's typeless ones: the name of a family
/// (another member of that family is what [`crate::resource::family`] gives
/// back for it). A view may not name one.
fn typeless(format: u32) -> bool {
    crate::resource::family(format) == format
        && (0..=200).any(|member| member != format && crate::resource::family(member) == format)
}

/// The format the pass's views must name over the image: the image's own,
/// unless that is typeless — then the concrete `named` format the encode
/// already reads it through, which is a member of the image's family (the
/// title made that view). `None` when neither is usable.
fn view_format(own: DXGI_FORMAT, named: Option<u32>) -> Option<DXGI_FORMAT> {
    if own == DXGI_FORMAT_UNKNOWN {
        return None;
    }
    if !typeless(own.0 as u32) {
        return Some(own);
    }
    named.map(|format| DXGI_FORMAT(format as i32))
}

fn make(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    view: DXGI_FORMAT,
    style: Style,
) -> windows::core::Result<Pass> {
    let source = match style {
        Style::Identity => SOURCE,
        Style::Curve => CURVE_SOURCE,
    };
    let knobs = match style {
        Style::Curve => curve(),
        Style::Identity => Knobs::default(),
    };
    let vs = compile(source, s!("vs"), s!("vs_5_1"))?;
    let ps = compile(source, s!("ps"), s!("ps_5_1"))?;
    let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
        pShaderBytecode: bytes.as_ptr().cast(),
        BytecodeLength: bytes.len(),
    };
    // SAFETY: the root signature is inside the vertex shader; plain
    // creations, and the description points at live data and never
    // releases its pointers.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs[..])?;
        // Every channel is written; no blending.
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = 0xf;
        let mut formats = [DXGI_FORMAT(0); 8];
        formats[0] = view;
        let pso: ID3D12PipelineState =
            device.CreateGraphicsPipelineState(&D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs),
                PS: code(&ps),
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // The pass writes colour and nothing else.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC::default(),
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                DSVFormat: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            })?;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let mut copy: Option<ID3D12Resource> = None;
        // Common at the start and at the end of every recording, as the
        // exposure pass's own copy is: every run records the same barriers.
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut copy,
        )?;
        let copy = copy.ok_or_else(windows::core::Error::empty)?;
        crate::host::name(&copy, || {
            format!("grade {} {width}x{height} f{}", style.name(), format.0)
        });
        let views: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?;
        let seen = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: view,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        device.CreateShaderResourceView(
            &copy,
            Some(&seen),
            views.GetCPUDescriptorHandleForHeapStart(),
        );
        Ok(Pass {
            style,
            knobs,
            root,
            pso,
            views,
            copy,
            width,
            height,
            format,
            view,
        })
    }
}

/// The image at `texture` (subresource 0), for the copy.
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

/// Records, in `list`, the pass `VICTORY_GRADE` asks for over `image`: the
/// image is copied into our own image, read from there and written back into
/// the image — as it is, or through the highlight curve. `format` is the
/// format the encode already reads the image through; it is used only when
/// the image's own is typeless. Returns whether anything was recorded (the
/// list's targets, descriptor heap, root signature and pipeline are then
/// ours, and the caller sets the title's again).
pub(crate) fn before_draw(list: &HostList, image: This, format: Option<u32>) -> bool {
    let Some(style) = style() else {
        return false;
    };
    let Some(host) = host() else {
        return false;
    };
    let Some(texture) = crate::resource::host_resource(image) else {
        return false;
    };
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || d.MipLevels != 1
        || d.DepthOrArraySize != 1
        || d.SampleDesc.Count != 1
        || d.Width == 0
        || d.Height == 0
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                width = d.Width,
                height = d.Height,
                format = d.Format.0,
                mips = d.MipLevels,
                layers = d.DepthOrArraySize,
                "no grade (VICTORY_GRADE): the colour it would read is not one image"
            )
        });
        return false;
    }
    // The pass writes the image itself, so the image must be one.
    if d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 == 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                format = d.Format.0,
                "no grade (VICTORY_GRADE): the colour it would write is not a render target"
            )
        });
        return false;
    }
    let Some(view) = view_format(d.Format, format) else {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                format = d.Format.0,
                named = format,
                "no grade (VICTORY_GRADE): the colour's format cannot be named by a view"
            )
        });
        return false;
    };
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    let Some(pass) = PASS
        .get_or_init(|| {
            make(
                &host.device,
                d.Width as u32,
                d.Height as u32,
                d.Format,
                view,
                style,
            )
            .inspect_err(|e| tracing::warn!(error = %e, "no grade (VICTORY_GRADE)"))
            .ok()
        })
        .as_ref()
    else {
        return false;
    };
    if (pass.width, pass.height, pass.format, pass.view)
        != (d.Width as u32, d.Height as u32, d.Format, view)
    {
        // The copy is made for one size and format; another colour image
        // would be written with the wrong texels.
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                width = d.Width,
                height = d.Height,
                format = d.Format.0,
                was = pass.width,
                "no grade (VICTORY_GRADE): the colour changed size or format"
            )
        });
        return false;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| match pass.style {
        Style::Identity => tracing::info!(
            width = pass.width,
            height = pass.height,
            format = pass.format.0,
            view = pass.view.0,
            "the finished colour is read and written back as it is (VICTORY_GRADE=identity)"
        ),
        Style::Curve => tracing::info!(
            width = pass.width,
            height = pass.height,
            format = pass.format.0,
            view = pass.view.0,
            knee = pass.knobs.knee,
            strength = pass.knobs.strength,
            white = pass.knobs.white,
            sat = pass.knobs.sat,
            "the finished colour gets the highlight curve (VICTORY_GRADE=curve)"
        ),
    });
    let l = &list.list;
    list.keep(&pass.copy);
    // The draw before left the image bound as a render target, and D3D12
    // copies neither into nor from a resource that is: the targets go
    // first. `crate::draw` sets the title's again when this returns.
    // SAFETY: an open list, and no view outlives the call.
    unsafe { l.OMSetRenderTargets(0, None, false, None) };
    // The image, where the copy reads it ...
    crate::list::ensure_state(list, image, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
    let viewport = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: pass.width as f32,
        Height: pass.height as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let whole = RECT {
        left: 0,
        top: 0,
        right: pass.width as i32,
        bottom: pass.height as i32,
    };
    // ... into our copy, which the draw then reads. The copy is common at
    // the start of every recording and left so at its end.
    // SAFETY: an open list and live textures of one size and format.
    unsafe {
        l.ResourceBarrier(&[crate::host::transition(
            &pass.copy,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )]);
        l.CopyTextureRegion(
            &subresource(&pass.copy),
            0,
            0,
            0,
            &subresource(&texture),
            None,
        );
        // The barrier into a shader-read state also orders the copy's
        // writes before the draw that reads them.
        l.ResourceBarrier(&[crate::host::transition(
            &pass.copy,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )]);
    }
    // The image is what the pass writes, and what the encode expects after
    // its last writer.
    crate::list::ensure_state(list, image, D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32);
    let written = D3D12_RENDER_TARGET_VIEW_DESC {
        Format: pass.view,
        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_RTV::default(),
        },
    };
    // A render-target view of the image lives only until this draw is
    // recorded (`Host::rtv`): the title's own targets are the caller's to
    // restore.
    let target = host.rtv(Some(&texture), Some(&written));
    // SAFETY: an open list; the pass's own heap, root signature and
    // pipeline; the colour in the render-target state as its target, our
    // copy in the shader-read one as its source.
    unsafe {
        l.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        l.SetGraphicsRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        l.OMSetRenderTargets(1, Some(&target as *const _), false, None);
        match pass.style {
            // The identity: the one table, no constant at all.
            Style::Identity => {
                l.SetGraphicsRootDescriptorTable(
                    0,
                    pass.views.GetGPUDescriptorHandleForHeapStart(),
                );
            }
            // The curve: b0 (knee, strength, white, sat) first, then the
            // table.
            Style::Curve => {
                let knobs = [
                    pass.knobs.knee,
                    pass.knobs.strength,
                    pass.knobs.white,
                    pass.knobs.sat,
                ];
                l.SetGraphicsRoot32BitConstants(0, 4, knobs.as_ptr().cast(), 0);
                l.SetGraphicsRootDescriptorTable(
                    1,
                    pass.views.GetGPUDescriptorHandleForHeapStart(),
                );
            }
        }
        // The input assembly is whatever the title's last draw left: these
        // three vertices are a triangle list (4, as `crate::draw` maps them).
        l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(4));
        crate::draw::viewports(list, &[viewport]);
        crate::draw::scissors(list, &[whole]);
        l.DrawInstanced(3, 1, 0, 0);
        // The draw has read our copy: only now may it rest in common again.
        l.ResourceBarrier(&[crate::host::transition(
            &pass.copy,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        )]);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::Style;

    /// `identity` and `curve` turn it on: anything else — absent, empty,
    /// off, a number or a spelling it does not know — leaves the picture
    /// alone.
    #[test]
    fn the_switch_is_identity_or_curve_and_nothing_else() {
        assert!(super::is_identity("identity"));
        assert!(super::is_identity(" identity "));
        assert!(super::is_identity("IDENTITY"));
        assert!(!super::is_identity("curve"));
        assert!(super::is_curve("curve"));
        assert!(super::is_curve(" CURVE "));
        assert!(super::is_curve("Curve"));
        assert!(!super::is_curve("identity"));
        for (set, wanted) in [
            ("identity", Some(Style::Identity)),
            (" Curve ", Some(Style::Curve)),
            ("", None),
            ("off", None),
            ("0", None),
            ("identities", None),
            ("curves", None),
        ] {
            assert_eq!(super::style_of(set), wanted, "{set:?}");
        }
    }

    /// The knobs are numbers inside their range and nothing else; the
    /// defaults are the ones the plan asks for.
    #[test]
    fn the_knobs_are_numbers_inside_their_range() {
        let knee = super::KNEE_RANGE;
        let strength = super::STRENGTH_RANGE;
        assert_eq!(super::KNEE, 0.75);
        assert_eq!(super::STRENGTH, 2.0);
        // Accepted, both ends included, spaces trimmed.
        assert_eq!(super::knob_of("0.75", knee), Some(0.75));
        assert_eq!(super::knob_of(" 0.3 ", knee), Some(0.3));
        assert_eq!(super::knob_of("0.95", knee), Some(0.95));
        assert_eq!(super::knob_of("0", strength), Some(0.0));
        assert_eq!(super::knob_of("20", strength), Some(20.0));
        // Not numbers, or outside the range: no value, the default stands.
        for bad in [
            "", " ", "knee", "0.2", "0.96", "1", "0", "inf", "-inf", "NaN",
        ] {
            assert_eq!(super::knob_of(bad, knee), None, "{bad:?} as a knee");
        }
        for bad in ["", "word", "-1", "20.1", "inf", "NaN"] {
            assert_eq!(super::knob_of(bad, strength), None, "{bad:?} as a strength");
        }
    }

    /// The shader's tone curve on one channel, mirrored here (in `f64`, for
    /// the step near the knee) so it can be checked without a GPU:
    /// `t = saturate((c - knee) / (1 - knee))` and
    /// `g = t + s t^2 (1 - t) / (1 + s t)`, written as
    /// `knee + (white - knee) g` above the knee and as `c` below it.
    fn tone(c: f64, knee: f64, strength: f64, white: f64) -> f64 {
        if c <= knee {
            return c;
        }
        let t = ((c - knee) / (1.0 - knee)).clamp(0.0, 1.0);
        let g = t + strength * t * t * (1.0 - t) / (1.0 + strength * t);
        knee + (white - knee) * g
    }

    /// The curve as it was before the white knob, `white = 1.0` made
    /// literal: the reference `tone(.., 1.0)` must be equal to, bit for bit.
    fn tone_before(c: f64, knee: f64, strength: f64) -> f64 {
        if c <= knee {
            return c;
        }
        let t = (c - knee) / (1.0 - knee);
        let g = t + strength * t * t * (1.0 - t) / (1.0 + strength * t);
        knee + (1.0 - knee) * g
    }

    /// The shader's croma, mirrored: `l = dot(rgb, luma)` and
    /// `rgb = lerp(rgb, l, 1 - sat)` per channel.
    fn croma(rgb: [f64; 3], sat: f64) -> [f64; 3] {
        let l = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        let w = 1.0 - sat;
        [
            rgb[0] + w * (l - rgb[0]),
            rgb[1] + w * (l - rgb[1]),
            rgb[2] + w * (l - rgb[2]),
        ]
    }

    /// `white = 1.0` is exactly the curve of before: the same values, not
    /// "close".
    #[test]
    fn white_one_is_the_curve_of_before() {
        for knee in [0.3f64, 0.5, 0.75, 0.95] {
            for strength in [0.0f64, 0.5, 2.0, 20.0] {
                for i in 0..=1000 {
                    let c = f64::from(i) / 1000.0;
                    assert_eq!(
                        tone(c, knee, strength, 1.0),
                        tone_before(c, knee, strength),
                        "c {c} knee {knee} strength {strength}"
                    );
                }
            }
        }
    }

    /// The two identities the curve must keep, and the slope at the knee:
    /// the knee does not move (from either side) and the top of the range is
    /// white; the slope is the identity's at `white = 1.0`; the curve never
    /// decreases, never leaves the picture, and never darkens.
    #[test]
    fn the_curve_keeps_the_knee_and_white_in_place() {
        for knee in [0.3f64, 0.5, 0.75, 0.95] {
            for strength in [0.0f64, 0.5, 2.0, 20.0] {
                assert!((tone(knee, knee, strength, 1.0) - knee).abs() < 1e-9);
                let below = knee - 1e-4;
                assert!((tone(below, knee, strength, 1.0) - below).abs() < 1e-9);
                assert!((tone(1.0, knee, strength, 1.0) - 1.0).abs() < 1e-9);
                // Soft knee: `g'(0) = 1`, so the knee has no crease. The
                // step is in `t`, which the knee's place cannot distort.
                let t = 1e-7;
                let slope = (tone(knee + t * (1.0 - knee), knee, strength, 1.0) - knee)
                    / (t * (1.0 - knee));
                assert!(
                    (slope - 1.0).abs() < 1e-4,
                    "knee {knee} strength {strength}: slope {slope}"
                );
                let mut last = 0.0f64;
                for i in 0..=1000 {
                    let c = f64::from(i) / 1000.0;
                    let out = tone(c, knee, strength, 1.0);
                    assert!(out >= last - 1e-9, "not monotone at {c} (knee {knee})");
                    assert!(out >= c - 1e-9, "darkened {c} (knee {knee})");
                    assert!(out <= 1.0 + 1e-9, "{out} above white (knee {knee})");
                    last = out;
                }
            }
        }
        // Strength 0 is the identity, everywhere.
        for i in 0..=1000 {
            let c = f64::from(i) / 1000.0;
            assert!((tone(c, 0.75, 0.0, 1.0) - c).abs() < 1e-12);
        }
    }

    /// `white < 1.0`: the top of the range lands exactly in `white`, the knee
    /// is still continuous, and the curve stays monotone for every allowed
    /// pair with `white > knee`.
    #[test]
    fn white_moves_the_top_and_the_curve_stays_monotone() {
        for knee in [0.3f64, 0.5, 0.75, 0.9, 0.95] {
            for white in [0.5f64, 0.6, 0.75, 0.9, 1.0] {
                if white <= knee {
                    continue;
                }
                for strength in [0.0f64, 0.5, 2.0, 20.0] {
                    // Continuity at the knee, from both sides.
                    assert!((tone(knee, knee, strength, white) - knee).abs() < 1e-9);
                    let below = knee - 1e-6;
                    assert!((tone(below, knee, strength, white) - below).abs() < 1e-9);
                    // The top of the range is exactly the white.
                    assert!(
                        (tone(1.0, knee, strength, white) - white).abs() < 1e-9,
                        "knee {knee} white {white} strength {strength}"
                    );
                    // Monotone, and never above the white.
                    let mut last = 0.0f64;
                    for i in 0..=2000 {
                        let c = f64::from(i) / 2000.0;
                        let out = tone(c, knee, strength, white);
                        assert!(
                            out >= last - 1e-9,
                            "not monotone at {c} (knee {knee} white {white})"
                        );
                        assert!(out <= white + 1e-9, "{out} above white {white}");
                        last = out;
                    }
                }
            }
        }
    }

    /// `sat = 1.0` is the **exact** identity (`==`, not "almost");
    /// `sat = 0.0` makes the three channels the luminance, which is a convex
    /// combination and so cannot create a new clip.
    #[test]
    fn the_croma_is_exact_at_one_and_grey_at_zero() {
        let rgb = [0.13f64, 0.71, 0.44];
        let same = croma(rgb, 1.0);
        assert!(same[0] == rgb[0] && same[1] == rgb[1] && same[2] == rgb[2]);
        let l = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        for channel in croma(rgb, 0.0) {
            assert!((channel - l).abs() < 1e-12, "{channel} vs luminance {l}");
        }
        let low = rgb.iter().copied().fold(f64::INFINITY, f64::min);
        let high = rgb.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(l >= low - 1e-12 && l <= high + 1e-12);
    }

    /// The two new knobs are numbers inside their range and nothing else; the
    /// defaults are 1.0, and a value outside the range is refused (the default
    /// is used and `knob` warns, never a silent clamp).
    #[test]
    fn the_white_and_sat_knobs_are_numbers_inside_their_range() {
        let white = super::WHITE_RANGE;
        let sat = super::SAT_RANGE;
        assert_eq!(super::WHITE, 1.0);
        assert_eq!(super::SAT, 1.0);
        // Accepted, both ends included, spaces trimmed.
        assert_eq!(super::knob_of("0.5", white), Some(0.5));
        assert_eq!(super::knob_of(" 1 ", white), Some(1.0));
        assert_eq!(super::knob_of("0", sat), Some(0.0));
        assert_eq!(super::knob_of("1.0", sat), Some(1.0));
        // Out of range or not a number: no value (the default stands, warned).
        for bad in ["", " ", "0.49", "1.01", "white", "inf", "NaN"] {
            assert_eq!(super::knob_of(bad, white), None, "{bad:?} as a white");
        }
        for bad in ["", " ", "-0.01", "1.01", "sat", "inf", "NaN"] {
            assert_eq!(super::knob_of(bad, sat), None, "{bad:?} as a sat");
        }
    }

    /// A white at or below the knee is not used: the shoulder would collapse
    /// or invert, so the knob falls back to 1.0 (with a warning), never a
    /// silent clamp to the knee.
    #[test]
    fn a_white_at_or_below_the_knee_falls_back_to_one() {
        assert_eq!(super::shoulder(0.75, 1.0), 1.0);
        assert_eq!(super::shoulder(0.75, 0.8), 0.8);
        assert_eq!(super::shoulder(0.75, 0.75), 1.0);
        assert_eq!(super::shoulder(0.5, 0.5), 1.0);
        assert_eq!(super::shoulder(0.95, 0.5), 1.0);
        assert_eq!(super::shoulder(0.3, 0.5), 0.5);
    }

    /// The views name a concrete format: the image's own when it is one; the
    /// encode's concrete `named` one only when the image is typeless, which
    /// no view may name.
    #[test]
    fn the_views_name_a_concrete_format() {
        use windows::Win32::Graphics::Dxgi::Common::*;
        // Concrete: its own name, whatever the encode says.
        for own in [
            DXGI_FORMAT_R10G10B10A2_UNORM,
            DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
        ] {
            assert!(!super::typeless(own.0 as u32), "{own:?}");
            assert_eq!(
                super::view_format(own, Some(DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32)),
                Some(own)
            );
            assert_eq!(super::view_format(own, None), Some(own));
        }
        // Typeless: the family's own name, and the encode's concrete one is
        // used; without it there is no view, so nothing is recorded.
        for own in [
            DXGI_FORMAT_R8G8B8A8_TYPELESS,
            DXGI_FORMAT_R10G10B10A2_TYPELESS,
            DXGI_FORMAT_R16G16B16A16_TYPELESS,
        ] {
            assert!(super::typeless(own.0 as u32), "{own:?}");
            assert_eq!(super::view_format(own, None), None);
        }
        assert_eq!(
            super::view_format(
                DXGI_FORMAT_R10G10B10A2_TYPELESS,
                Some(DXGI_FORMAT_R10G10B10A2_UNORM.0 as u32)
            ),
            Some(DXGI_FORMAT_R10G10B10A2_UNORM)
        );
        assert_eq!(super::view_format(DXGI_FORMAT_UNKNOWN, Some(28)), None);
    }

    /// The vertex program's triangle covers the whole viewport and every
    /// pixel reads its own texel: `p` is (0,0), (2,0), (0,2) and `at` is
    /// `p * 2 - 1`, so the three vertices are (-1,-1), (3,-1) and (-1,3),
    /// which hold the whole clip square.
    #[test]
    fn the_triangle_covers_the_whole_image() {
        // The vertex program, mirrored.
        let at: Vec<(f32, f32)> = (0..3u32)
            .map(|id| {
                let p = (
                    ((id << 1) & 2) as f32,
                    (id & 2) as f32,
                );
                (p.0 * 2.0 - 1.0, p.1 * 2.0 - 1.0)
            })
            .collect();
        assert_eq!(at, vec![(-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)]);
        // The edge functions of the triangle: all of one sign inside it.
        let edge = |a: (f32, f32), b: (f32, f32), q: (f32, f32)| {
            (b.0 - a.0) * (q.1 - a.1) - (b.1 - a.1) * (q.0 - a.0)
        };
        let inside = |q: (f32, f32)| {
            [
                edge(at[0], at[1], q),
                edge(at[1], at[2], q),
                edge(at[2], at[0], q),
            ]
            .iter()
            .all(|&w| w >= 0.0)
        };
        // Every corner of the viewport, and its centre, are covered.
        for q in [(-1.0f32, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (0.0, 0.0)] {
            assert!(inside(q), "{q:?} outside the triangle");
        }
    }

    /// Both passes compile with the system's compiler and the host takes
    /// them (small copies, so the tests need no 1920x1080 allocation). The
    /// host is absent outside the game, where the test returns and the
    /// shaders are checked by the compiler alone.
    #[test]
    fn the_passes_build() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R10G10B10A2_UNORM;
        for style in [Style::Identity, Style::Curve] {
            super::make(&host.device, 16, 16, format, format, style)
                .unwrap_or_else(|e| panic!("the {style:?} pass: {e}"));
        }
    }
}
