//! Opt-in switches inside the title's own code, for known builds.
//!
//! Only per-build offsets and short byte signatures live here (asset policy:
//! nothing of the title itself). The build hash is not needed and not
//! checked: every patch compares the exact bytes it would replace and
//! changes nothing when they differ, so a different build is a no-op.
//!
//! Three levers, each off unless its environment variable says otherwise:
//! `VICTORY_NO_WATCHDOG` disarms the title's watchdog, `VICTORY_OWN_UPSCALER`
//! makes it use its own temporal upscaler on every profile, and
//! `VICTORY_RENDER_SCALE` (0.2..=1.0) fixes its internal render scale.

use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// Environment variable that disarms the watchdog (`1`/anything but `0`).
const NO_WATCHDOG_ENV: &str = "VICTORY_NO_WATCHDOG";
/// Environment variable that forces the title's own upscaler.
const OWN_UPSCALER_ENV: &str = "VICTORY_OWN_UPSCALER";
/// Environment variable that pins the internal render scale.
const RENDER_SCALE_ENV: &str = "VICTORY_RENDER_SCALE";

/// One code patch: `rva` must hold exactly `expect` before `new` is written.
struct Patch<'a> {
    what: &'static str,
    rva: usize,
    expect: &'a [u8],
    new: &'a [u8],
}

/// A patch whose bytes are computed at load time.
struct OwnedPatch {
    what: &'static str,
    rva: usize,
    expect: Vec<u8>,
    new: Vec<u8>,
}

/// The title's watchdog thread ends the process on purpose when its main
/// thread has not advanced for about 30 one-second ticks. Two places do it:
/// the handler it calls (`mov dword ptr [0], imm32`) and, when the handler
/// returns or is unset, a fallback right after the call (`mov dword ptr [0],
/// 0; int3`). Jumping over the whole block returns the loop to its normal
/// path; the second patch makes the handler harmless if something else calls
/// it.
const WATCHDOG: &[Patch<'static>] = &[
    Patch {
        what: "watchdog: skip the call and the fallback fault",
        rva: 0x2215c27,
        // mov rax, [rip + disp32] (the handler pointer)
        expect: &[0x48, 0x8B, 0x05, 0x1A, 0x07, 0xFA, 0x04],
        // jmp +0x18 (to the loop's normal continuation), then NOPs
        new: &[0xEB, 0x18, 0x90, 0x90, 0x90, 0x90, 0x90],
    },
    Patch {
        what: "watchdog: handler's deliberate fault",
        rva: 0x23cf683,
        expect: &[
            0xC7, 0x04, 0x25, 0, 0, 0, 0, 0x21, 0xD4, 0x8B, 0x4D,
        ],
        // an eleven-byte NOP
        new: &[0x66, 0x66, 0x2E, 0x0F, 0x1F, 0x84, 0, 0, 0, 0, 0],
    },
];

/// The upscaler choice: `mov eax, 6` (the title's own) followed by
/// `cmovne eax, ecx` (FSR when the profile asks for it). NOPing the `cmovne`
/// keeps 6, so every profile, Series S included, takes the title's upscaler
/// and not FSR, which does not pass through our translated shaders.
const OWN_UPSCALER: Patch<'static> = Patch {
    what: "upscaler: always the title's own",
    rva: 0x7e9426,
    expect: &[0x0F, 0x45, 0xC1],
    new: &[0x0F, 0x1F, 0x00],
};

/// Read-only float constants of this build usable as the FSR scale: value,
/// RVA.
const FSR_SCALES: [(f32, usize); 4] = [
    (0.5, 0x5fc6edc),
    (0.4, 0x5fc6f90),
    (1.0 / 3.0, 0x615a240),
    (0.75, 0x6159b04),
];

/// The render-scale patches for `scale`, with the image at `base`:
/// - the own upscaler's minimum scale (an immediate, 2/3 by default) set to
///   `scale` and its maximum made equal to the minimum (`vmaxss` to
///   `vmovss`), so the scale is fixed;
/// - FSR's scale (a load of 2/3 on Series S) pointed at a constant equal to
///   `scale`, when this build has one, and its maximum size made the scaled
///   size (vtable slot 11 to slot 10's function).
fn render_scale_patches(base: usize, scale: f32, fsr: bool) -> Vec<OwnedPatch> {
    let mut patches = vec![
        OwnedPatch {
            what: "render scale: own upscaler minimum",
            rva: 0x28f1e7e,
            expect: vec![0xAB, 0xAA, 0x2A, 0x3F],
            new: scale.to_le_bytes().to_vec(),
        },
        OwnedPatch {
            what: "render scale: own upscaler maximum = minimum",
            rva: 0x28f1e87,
            expect: vec![0xC5, 0xFA, 0x5F, 0x86, 0x90, 0, 0, 0],
            new: vec![0xC5, 0xFA, 0x10, 0x86, 0x90, 0, 0, 0],
        },
    ];
    if !fsr {
        return patches;
    }
    match FSR_SCALES.iter().find(|(v, _)| (v - scale).abs() < 1e-3) {
        Some(&(_, constant)) => {
            // vmovss xmm0, [rip + disp32]; the next instruction is at +8.
            const LOAD: usize = 0x28ef6b2;
            let disp = (constant as i64 - (LOAD as i64 + 8)) as i32;
            let mut new = vec![0xC5, 0xFA, 0x10, 0x05];
            new.extend_from_slice(&disp.to_le_bytes());
            patches.push(OwnedPatch {
                what: "render scale: FSR scale",
                rva: LOAD,
                expect: vec![0xC5, 0xFA, 0x10, 0x05, 0xA2, 0xE2, 0x86, 0x03],
                new,
            });
            patches.push(OwnedPatch {
                what: "render scale: FSR maximum = scaled size",
                rva: 0x6300ba8,
                expect: ((base + 0x28ef6f0) as u64).to_le_bytes().to_vec(),
                new: ((base + 0x28ef680) as u64).to_le_bytes().to_vec(),
            });
        }
        None => tracing::warn!(
            scale,
            var = RENDER_SCALE_ENV,
            "no FSR constant for this scale (0.5, 0.4, 1/3, 0.75); FSR keeps its own (use VICTORY_OWN_UPSCALER=1)"
        ),
    }
    patches
}

/// The scale `VICTORY_RENDER_SCALE` asks for, if valid.
fn parse_scale(text: &str) -> Option<f32> {
    let v: f32 = text.trim().parse().ok()?;
    (0.2..=1.0).contains(&v).then_some(v)
}

/// Whether `var` is set to something other than empty or `0`.
fn enabled(var: &str) -> bool {
    std::env::var_os(var).is_some_and(|v| !v.is_empty() && v.to_string_lossy().as_ref() != "0")
}

/// Applies the levers the environment asks for: the watchdog, the title's
/// own upscaler, and the fixed render scale. Returns how many patches
/// applied; zero when no variable is set, which is the default.
///
/// Call it once on a freshly mapped image, before any of its code runs:
/// `base` is the address `load` returned, not a preferred or guessed one.
/// Each patch is applied only if the bytes at `base + rva` are exactly the
/// expected ones, so another build of the title is left alone.
pub fn resolution_levers(base: usize) -> usize {
    let scale = std::env::var(RENDER_SCALE_ENV).ok();
    let own = enabled(OWN_UPSCALER_ENV);
    let watchdog = enabled(NO_WATCHDOG_ENV);
    if scale.is_none() && !own && !watchdog {
        return 0;
    }
    let mut applied = 0;
    if watchdog {
        for p in WATCHDOG {
            // SAFETY: `base` is the mapped image and no title code runs yet.
            applied += usize::from(unsafe { apply(base, p.what, p.rva, p.expect, p.new) });
        }
    }
    if own {
        // SAFETY: as above.
        applied += usize::from(unsafe {
            apply(
                base,
                OWN_UPSCALER.what,
                OWN_UPSCALER.rva,
                OWN_UPSCALER.expect,
                OWN_UPSCALER.new,
            )
        });
    }
    if let Some(text) = scale {
        match parse_scale(&text) {
            Some(scale) => {
                // With the own upscaler forced, FSR is left as it is.
                for p in render_scale_patches(base, scale, !own) {
                    // SAFETY: as above.
                    applied += usize::from(unsafe { apply(base, p.what, p.rva, &p.expect, &p.new) });
                }
            }
            None => tracing::warn!(
                var = RENDER_SCALE_ENV,
                text,
                "wants a number in 0.2..=1.0; nothing patched"
            ),
        }
    }
    tracing::info!(applied, "resolution levers applied");
    applied
}

/// Writes `new` over `base + rva` if those bytes hold exactly `expect`.
///
/// # Safety
/// `base` is the image mapped by [`crate::load`], `base + rva` is inside it,
/// and no code of it is running (the patches change instructions).
unsafe fn apply(base: usize, what: &str, rva: usize, expect: &[u8], new: &[u8]) -> bool {
    if new.len() != expect.len() || expect.is_empty() {
        tracing::warn!(what, "hook skipped: the replacement is not the same size");
        return false;
    }
    let at = (base + rva) as *const u8;
    // SAFETY: forwarded from this function's contract.
    let now = unsafe { std::slice::from_raw_parts(at, expect.len()) };
    if now != expect {
        tracing::warn!(what, "hook skipped: the bytes are not the expected ones");
        return false;
    }
    let mut old = 0;
    // SAFETY: changing the protection of those bytes only, restored below.
    if unsafe { VirtualProtect(at as _, new.len(), PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
        tracing::warn!(
            what,
            error = %std::io::Error::last_os_error(),
            "hook skipped: VirtualProtect failed"
        );
        return false;
    }
    // SAFETY: writable now, and the bytes are the ones checked above.
    unsafe {
        std::ptr::copy_nonoverlapping(new.as_ptr(), at as *mut u8, new.len());
        VirtualProtect(at as _, new.len(), old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), at as _, new.len());
    }
    tracing::info!(what, "hook applied");
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_outside_the_floor_are_refused() {
        assert_eq!(parse_scale("0.5"), Some(0.5));
        assert_eq!(parse_scale(" 1 "), Some(1.0));
        assert_eq!(parse_scale("0.2"), Some(0.2));
        assert_eq!(parse_scale("0.1"), None);
        assert_eq!(parse_scale("1.5"), None);
        assert_eq!(parse_scale("half"), None);
    }

    #[test]
    fn half_scale_patches_both_upscalers() {
        let base = 0x7ff0_0000_0000;
        let p = render_scale_patches(base, 0.5, true);
        assert_eq!(p.len(), 4);
        assert_eq!(p[0].new, 0.5f32.to_le_bytes());
        // The FSR load now reads 0.5 at 0x5fc6edc: disp from the next instruction.
        assert_eq!(p[2].new, [0xC5, 0xFA, 0x10, 0x05, 0x22, 0x78, 0x6D, 0x03]);
        assert_eq!(p[3].new, (base as u64 + 0x28ef680).to_le_bytes());
        assert!(p.iter().all(|p| p.new.len() == p.expect.len()));
    }

    #[test]
    fn odd_scales_patch_only_the_own_upscaler() {
        assert_eq!(render_scale_patches(0, 0.6, true).len(), 2);
        assert_eq!(render_scale_patches(0, 0.5, false).len(), 2);
    }

    #[test]
    fn no_variable_means_nothing_to_do() {
        // The lever list is empty only when no variable asks for it; the
        // patch tables themselves are constant, so test the gate directly.
        assert!(!enabled("VICTORY_NOTHING_SETS_THIS_VARIABLE"));
    }
}
