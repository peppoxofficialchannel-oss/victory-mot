//! Which host GPU the host D3D12 device runs on.
//!
//! A machine with two graphics adapters (a laptop with an integrated one and
//! a discrete one, say) gives a device created without an adapter to the one
//! DXGI lists first, which is the integrated one: pipelines made for the
//! discrete card are then refused there with `E_INVALIDARG` ("Parametro non
//! corretto") and every measurement is of the wrong card. This picker takes
//! the hardware adapter (never `DXGI_ADAPTER_FLAG_SOFTWARE`) with the most
//! `DedicatedVideoMemory`: the discrete card where there is one, and the only
//! card where there is one adapter, so nothing changes there.
//!
//! `VICTORY_ADAPTER=<index|name substring>` names the adapter instead (the
//! index is the one DXGI enumerates it at, the substring is matched without
//! case against its description). An override that matches no hardware
//! adapter is a warning and the rule applies.
//!
//! The choice is logged once per process: without that line no run says which
//! card it measured.

use std::sync::Once;

use windows::Win32::Foundation::LUID;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

/// The adapter the host device is created on, and why.
pub struct Chosen {
    pub adapter: IDXGIAdapter3,
    /// `DXGI_ADAPTER_DESC1::Description`, up to the first NUL.
    pub name: String,
    pub luid: LUID,
    /// `DXGI_ADAPTER_DESC1::DedicatedVideoMemory`, in bytes.
    pub dedicated: usize,
    /// `DXGI_ADAPTER_DESC1::SharedSystemMemory`, in bytes.
    pub shared: usize,
    /// It is the first hardware adapter DXGI lists: the one a device created
    /// without an adapter gets, so the choice changes nothing.
    pub preferred: bool,
    /// `VICTORY_ADAPTER` named it.
    pub by_override: bool,
}

/// Said once per process, whichever path picks first (the layer's host device
/// and the tests' `Host::new` share this picker).
static SAID: Once = Once::new();

fn say(text: &str) {
    SAID.call_once(|| tracing::info!("{text}"));
}

fn description(desc: &DXGI_ADAPTER_DESC1) -> String {
    let units = &desc.Description;
    let end = units.iter().position(|&u| u == 0).unwrap_or(units.len());
    String::from_utf16_lossy(&units[..end]).trim().to_owned()
}

/// `VICTORY_ADAPTER`, trimmed; `None` when unset or empty.
fn asked() -> Option<String> {
    std::env::var("VICTORY_ADAPTER")
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
}

impl Chosen {
    /// The line that says which card the process measures, once.
    fn declare(&self) {
        say(&format!(
            "host GPU adapter: {} (LUID {:08x}:{:08x}), {} MB dedicated, {} MB shared, preferred={} ({})",
            self.name,
            self.luid.HighPart as u32,
            self.luid.LowPart,
            self.dedicated >> 20,
            self.shared >> 20,
            self.preferred,
            if self.by_override {
                "VICTORY_ADAPTER"
            } else {
                "most dedicated memory"
            },
        ));
    }
}

/// The adapter to create the host device on: the hardware one with the most
/// dedicated memory, or the one `VICTORY_ADAPTER` names.
///
/// `None` when no hardware adapter can be enumerated (a software-only
/// machine): the caller then creates the device without an adapter, as
/// before.
pub fn choose() -> Option<Chosen> {
    // SAFETY: plain DXGI enumeration and adapter description queries; every
    // interface comes from the factory and outlives the call.
    unsafe {
        let factory: IDXGIFactory4 = match CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) {
            Ok(factory) => factory,
            Err(e) => {
                say(&format!(
                    "host GPU adapter: DXGI factory unavailable ({e}); the device gets the default adapter"
                ));
                return None;
            }
        };
        let mut adapters: Vec<(u32, IDXGIAdapter3, DXGI_ADAPTER_DESC1)> = Vec::new();
        let mut index = 0;
        while let Ok(adapter) = factory.EnumAdapters1(index) {
            if let (Ok(desc), Ok(threes)) = (adapter.GetDesc1(), adapter.cast::<IDXGIAdapter3>())
                && desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0
            {
                adapters.push((index, threes, desc));
            }
            index += 1;
        }
        let Some(&(first, ..)) = adapters.first() else {
            say("host GPU adapter: no hardware adapter is enumerable; the device gets the default adapter");
            return None;
        };

        let asked = asked();
        let (at, by_override) = match &asked {
            Some(ask) => {
                let found = ask
                    .parse::<u32>()
                    .ok()
                    .and_then(|want| adapters.iter().position(|&(index, ..)| index == want))
                    .or_else(|| {
                        let want = ask.to_lowercase();
                        adapters
                            .iter()
                            .position(|(_, _, desc)| description(desc).to_lowercase().contains(&want))
                    });
                match found {
                    Some(at) => (at, true),
                    None => {
                        tracing::warn!(
                            asked = %ask,
                            "VICTORY_ADAPTER names no hardware adapter: the most dedicated memory is taken instead"
                        );
                        (most_dedicated(&adapters), false)
                    }
                }
            }
            None => (most_dedicated(&adapters), false),
        };

        let (index, adapter, desc) = adapters.swap_remove(at);
        let chosen = Chosen {
            adapter,
            name: description(&desc),
            luid: desc.AdapterLuid,
            dedicated: desc.DedicatedVideoMemory,
            shared: desc.SharedSystemMemory,
            preferred: index == first,
            by_override,
        };
        chosen.declare();
        Some(chosen)
    }
}

/// The index of the adapter with the most dedicated memory (the first one on
/// a tie).
fn most_dedicated(adapters: &[(u32, IDXGIAdapter3, DXGI_ADAPTER_DESC1)]) -> usize {
    let mut at = 0;
    for (i, (_, _, desc)) in adapters.iter().enumerate() {
        if desc.DedicatedVideoMemory > adapters[at].2.DedicatedVideoMemory {
            at = i;
        }
    }
    at
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the picker names on this machine: a hardware adapter, whose
    /// description is the one reported (`cargo test ... -- --nocapture`).
    /// Only DXGI queries: no device is made.
    #[test]
    fn the_pick_is_a_hardware_adapter() {
        let Some(chosen) = choose() else {
            eprintln!("host GPU adapter: no hardware adapter to pick");
            return;
        };
        eprintln!(
            "host GPU adapter: {} (LUID {:08x}:{:08x}), {} MB dedicated, {} MB shared, preferred={}",
            chosen.name,
            chosen.luid.HighPart as u32,
            chosen.luid.LowPart,
            chosen.dedicated >> 20,
            chosen.shared >> 20,
            chosen.preferred,
        );
        // SAFETY: a plain description query on an adapter we hold.
        let desc = unsafe { chosen.adapter.GetDesc1() }.expect("adapter description");
        assert_eq!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32, 0);
        assert_eq!(desc.DedicatedVideoMemory, chosen.dedicated);
    }
}
