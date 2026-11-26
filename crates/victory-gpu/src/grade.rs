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
//! So the draw reads the title's image (`SRV`) and writes **our copy** of
//! it (`RTV`, made once, of the same size and format); then one
//! `CopyTextureRegion` puts the copy back. Both the title's targets and our
//! copy are let go before the copy runs, and `crate::draw`'s `replay` sets
//! the title's targets, viewports and scissors again as soon as this
//! module returns.
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

/// The one value `VICTORY_GRADE` takes for now: the pass that leaves the
/// picture as it is.
const IDENTITY: &str = "identity";

/// Whether the identity pass is asked for (`VICTORY_GRADE=identity`, any
/// case, spaces trimmed). Absent or anything else: off.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_GRADE").is_ok_and(|set| is_identity(&set)))
}

/// Whether `set` (what the variable holds) asks for the identity.
fn is_identity(set: &str) -> bool {
    set.trim().eq_ignore_ascii_case(IDENTITY)
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

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// The shader-visible heap holding the one view the pass reads.
    views: ID3D12DescriptorHeap,
    /// The heap holding the one view it writes.
    targets: ID3D12DescriptorHeap,
    /// Our copy of the colour: written by the pass, then copied back. Kept
    /// in the render-target state between runs.
    copy: ID3D12Resource,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

/// One shader entry point of [`SOURCE`], compiled.
fn compile(
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> windows::core::Result<Vec<u8>> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let made = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
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
                "the identity pass of the grade does not compile: {}",
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

fn make(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
) -> windows::core::Result<Pass> {
    let vs = compile(s!("vs"), s!("vs_5_1"))?;
    let ps = compile(s!("ps"), s!("ps_5_1"))?;
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
        formats[0] = format;
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
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
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
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            None,
            &mut copy,
        )?;
        let copy = copy.ok_or_else(windows::core::Error::empty)?;
        crate::host::name(&copy, || {
            format!("grade identity {width}x{height} f{}", format.0)
        });
        let views: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?;
        let seen = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
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
        let targets: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })?;
        let written = D3D12_RENDER_TARGET_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_RTV::default(),
            },
        };
        device.CreateRenderTargetView(
            &copy,
            Some(&written),
            targets.GetCPUDescriptorHandleForHeapStart(),
        );
        Ok(Pass {
            root,
            pso,
            views,
            targets,
            copy,
            width,
            height,
            format,
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

/// Records, in `list`, the identity pass over `image`: the image is read
/// through `format` (`None`: the resource's own) and written back as it is.
/// Returns whether anything was recorded (the list's targets, descriptor
/// heap, root signature and pipeline are then ours, and the caller sets the
/// title's again).
pub(crate) fn before_draw(list: &HostList, image: This, format: Option<u32>) -> bool {
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
    let format = DXGI_FORMAT(format.unwrap_or(d.Format.0 as u32) as i32);
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    let Some(pass) = PASS
        .get_or_init(|| {
            make(&host.device, d.Width as u32, d.Height as u32, format)
                .inspect_err(|e| tracing::warn!(error = %e, "no grade (VICTORY_GRADE)"))
                .ok()
        })
        .as_ref()
    else {
        return false;
    };
    if (pass.width, pass.height, pass.format) != (d.Width as u32, d.Height as u32, format) {
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
    ONCE.call_once(|| {
        tracing::info!(
            width = pass.width,
            height = pass.height,
            format = pass.format.0,
            "the finished colour is read and written back as it is (VICTORY_GRADE=identity)"
        )
    });
    let l = &list.list;
    list.keep(&pass.copy);
    // The draw before left the image bound as a render target, and D3D12
    // copies neither into nor from a resource that is: the targets go
    // first. `crate::draw` sets the title's again when this returns.
    // SAFETY: an open list, and no view outlives the call.
    unsafe { l.OMSetRenderTargets(0, None, false, None) };
    // The colour, where the draw reads it.
    crate::list::ensure_state(
        list,
        image,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 as u32,
    );
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
    // SAFETY: an open list; the pass's own heap, root signature and
    // pipeline; the colour in the shader-read state, our copy in the
    // render-target one.
    unsafe {
        l.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        l.SetGraphicsRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        let target = pass.targets.GetCPUDescriptorHandleForHeapStart();
        l.OMSetRenderTargets(1, Some(&target as *const _), false, None);
        l.SetGraphicsRootDescriptorTable(0, pass.views.GetGPUDescriptorHandleForHeapStart());
        // The input assembly is whatever the title's last draw left: these
        // three vertices are a triangle list (4, as `crate::draw` maps them).
        l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(4));
        crate::draw::viewports(list, &[viewport]);
        crate::draw::scissors(list, &[whole]);
        l.DrawInstanced(3, 1, 0, 0);
        // Our copy is still bound as a target: let it go before reading it.
        l.OMSetRenderTargets(0, None, false, None);
        l.ResourceBarrier(&[crate::host::transition(
            &pass.copy,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
    }
    crate::list::ensure_state(list, image, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32);
    // SAFETY: an open list and live textures of one size and format.
    unsafe {
        l.CopyTextureRegion(
            &subresource(&texture),
            0,
            0,
            0,
            &subresource(&pass.copy),
            None,
        )
    };
    // Back to the state the title's last writer left it in, for the encode.
    crate::list::ensure_state(list, image, D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32);
    // SAFETY: as above; our copy stays in the render-target state between
    // runs, so every run records the same barriers.
    unsafe {
        l.ResourceBarrier(&[crate::host::transition(
            &pass.copy,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        )])
    };
    true
}

#[cfg(test)]
mod tests {
    /// Only `identity` turns it on: absent, empty, off, a number or a
    /// spelling it does not know leave the picture alone.
    #[test]
    fn the_switch_is_identity_and_nothing_else() {
        assert!(super::is_identity("identity"));
        assert!(super::is_identity(" identity "));
        assert!(super::is_identity("IDENTITY"));
        assert!(!super::is_identity(""));
        assert!(!super::is_identity("off"));
        assert!(!super::is_identity("0"));
        assert!(!super::is_identity("1"));
        assert!(!super::is_identity("identities"));
        assert!(!super::is_identity("curve"));
    }

    /// The pass compiles with the system's compiler and the host takes it
    /// (a small copy, so the test needs no 1920x1080 allocation).
    #[test]
    fn identity_pass_builds() {
        let Some(host) = crate::host::host() else {
            return;
        };
        super::make(
            &host.device,
            16,
            16,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R10G10B10A2_UNORM,
        )
        .expect("the identity pass");
    }
}
