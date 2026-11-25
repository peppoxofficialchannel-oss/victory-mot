//! The console's ray-tracing structure builder (`XboxBvhBuilder.dll`), from the user's own
//! copy (`VICTORY_TEST_BVH_DLL=<path>`; skipped when unset, as the asset policy asks): its
//! wave64 programs on a host of 32-lane waves give the same bytes in the form the translator
//! picks for them and in the paired form (one host wave per console wave, each thread two
//! of its lanes: every cross-lane operation sees all 64 lanes), on synthetic inputs. Run on
//! the host GPU when it takes waves of 32. Asserts a property (the two forms agree), never
//! the title's data.

use super::host::{Dispatch, Host, Root};
use super::{HostShader, lower};
use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType, StageIo};
use crate::{io, resource, structure};

/// The DXBC containers in a file: each found by its magic and the size its header gives.
fn containers(bytes: &[u8]) -> Vec<&[u8]> {
    let mut found = Vec::new();
    let mut at = 0;
    while let Some(i) = bytes[at..].windows(4).position(|w| w == b"DXBC") {
        let start = at + i;
        let size = bytes
            .get(start + 24..start + 28)
            .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize);
        if size >= 64 && start + size <= bytes.len() {
            found.push(&bytes[start..start + size]);
            at = start + size;
        } else {
            at = start + 4;
        }
    }
    found
}

/// A small generator of test inputs (xorshift), so a failure can be run again.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

/// The two forms of `native` on a host of 32-lane waves: the translator's pick and the
/// paired one. None if either does not translate, or the pick is the paired form already.
fn forms(c: &Container, native: &NativeProgram) -> Option<(HostShader, HostShader)> {
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let r = (|| {
        let p = crate::spill::build(native).ok()?;
        let st = structure::structure(&p).ok()?;
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).ok()?;
        let io = io::raise(&p, native).ok()?;
        let picked = super::emit(c, native, &p, &st, &io, &res).ok()?;
        let (paired, _) = super::paired(c, native, &p, &io, &res).ok()?;
        (picked.dxil != paired.dxil).then_some((picked, paired))
    })();
    lower::TEST_HOST_WAVE.set(was);
    r
}

/// Runs `shader` with root constants `constants` (per title root slot) and a root UAV per
/// title root descriptor holding `views[slot]`; returns the root UAVs' bytes as dwords.
fn run(
    host: &Host,
    shader: &HostShader,
    constants: &[(u8, Vec<u32>)],
    views: &[(u8, Vec<u32>)],
    groups: u32,
) -> Result<Vec<Vec<u32>>, String> {
    let (root, pso) = host
        .compute_pipeline_bounded(shader)
        .map_err(|e| format!("pipeline: {e}"))?;
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match u.root {
            RootType::Constants => constants
                .iter()
                .find(|c| c.0 == u.slot)
                .and_then(|c| c.1.get(u.dword as usize).copied())
                .unwrap_or(0),
            // (views begin on their address: no bias)
            _ => 0,
        };
    }
    let mut roots = vec![(0, Root::Constants(ud))];
    // as `Host::root_signature_for` orders them: by slot, then CBV, SRV, UAV
    let mut slots: Vec<(u8, u8)> = shader
        .root_cbvs
        .iter()
        .map(|&s| (s, 0))
        .chain(shader.root_srvs.iter().map(|&s| (s, 1)))
        .chain(shader.root_uavs.iter().map(|&s| (s, 2)))
        .collect();
    slots.sort_unstable();
    slots.dedup();
    // `VICTORY_TEST_BVH_IN_PLACE=1`: slot 2 is slot 1's memory, as the title rehydrates
    // its structures in place (source and destination the same address)
    let in_place = std::env::var_os("VICTORY_TEST_BVH_IN_PLACE").is_some_and(|v| v == "1");
    let mut param_of_slot1 = None;
    for (n, (slot, kind)) in slots.into_iter().enumerate() {
        let data = views
            .iter()
            .find(|v| v.0 == slot)
            .map_or_else(|| vec![0; 64], |v| v.1.clone());
        let param = n as u32 + 1;
        if kind == 2 && slot == 1 {
            param_of_slot1 = Some(param);
        }
        roots.push((
            param,
            match kind {
                2 if in_place && slot == 2 && param_of_slot1.is_some() => {
                    Root::UavAlias(param_of_slot1.unwrap_or(0))
                }
                2 => Root::Uav(data),
                1 => Root::View { srv: true, data },
                _ => Root::View { srv: false, data },
            },
        ));
    }
    host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots,
            views: &[],
            samplers: &[],
            groups: [groups, 1, 1],
        },
    )
    .map_err(|e| format!("dispatch: {e}"))
}

/// Box rehydration (both node sizes): 64 nodes a console wave, each lane reading a control
/// word whose top byte counts the dwords of its payload and whose low 24 bits mark children,
/// the payload found by the prefix sums of those over the wave's 64 lanes (across its
/// halves). Inputs: control words with small counts, random payloads, a destination with
/// room for the nodes and their parent records.
#[test]
fn the_builders_wave64_programs_agree_with_their_paired_form() {
    let Some(path) = std::env::var_os("VICTORY_TEST_BVH_DLL") else {
        eprintln!("VICTORY_TEST_BVH_DLL not set; not run");
        return;
    };
    let bytes = std::fs::read(&path).expect("the builder DLL");
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let mut compared = 0;
    let mut differ = Vec::new();
    // `VICTORY_TEST_BVH_ONLY=<n>`: container n alone; `VICTORY_TEST_BVH_FORM=picked|paired`:
    // that form alone (its digest printed), to tell which one stops the device.
    let only: Option<usize> = std::env::var("VICTORY_TEST_BVH_ONLY")
        .ok()
        .and_then(|v| v.parse().ok());
    let form = std::env::var("VICTORY_TEST_BVH_FORM").unwrap_or_default();
    for (k, container) in containers(&bytes).into_iter().enumerate() {
        if only.is_some_and(|o| o != k) {
            continue;
        }
        let Ok(c) = Container::parse(container) else {
            continue;
        };
        let Some(part) = c.part(FourCC::DXIL) else {
            continue;
        };
        let Ok(programs) = crate::native_programs(part.data) else {
            continue;
        };
        for native in programs {
            let StageIo::Cs(cs) = native.io else { continue };
            if native.stage != HwStage::Cs || !native.wave64 {
                continue;
            }
            let Some((picked, paired)) = forms(&c, &native) else {
                continue;
            };
            // Programs that read title memory by address need its directory: not here.
            if picked.root_srvs.contains(&super::SLOT_MEMORY) {
                continue;
            }
            let threads: u32 = cs.threads.iter().product();
            for seed in 1..=4u64 {
                let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed);
                // root constants of slot 0: base offset 0, parent records from 128 KiB on,
                // child pointers from 0, a flag set; the rest small numbers
                let mut constants = vec![0u32; 19];
                constants[1] = 0x2_0000;
                constants[3] = 1;
                for c in constants.iter_mut().skip(4) {
                    *c = rng.next() % 64;
                }
                // source: per group 8 KiB of control words then payloads
                let mut src = vec![0u32; 0x1_0000 / 4];
                for (i, w) in src.iter_mut().enumerate() {
                    *w = if i % 2048 < 64 {
                        (rng.next() % 4) << 24 | (rng.next() & 0x00FF_FFFF)
                    } else {
                        rng.next()
                    };
                }
                let dst = vec![0u32; 0x8_0000 / 4];
                // keys and the like for the programs whose slot 0 is read too; for the
                // radix sort's scatter (slots 0 to 5): keys, values (their indices), the
                // first place of each digit of the keys' low byte, block sums (none), keys
                // and values out
                let keys: Vec<u32> = (0..0x1_0000 / 4).map(|_| rng.next()).collect();
                let mut firsts = vec![0u32; 0x1_0000 / 4];
                let mut count = [0u32; 256];
                for &key in &keys[..2048] {
                    count[(key & 0xFF) as usize] += 1;
                }
                let mut at = 0;
                for (d, n) in count.iter().enumerate() {
                    firsts[d] = at;
                    at += n;
                }
                // (the radix sort's scatter is the program with seven root parameters)
                let radix = native.layout.len() == 7;
                let views = [
                    (0u8, keys.clone()),
                    (
                        1u8,
                        if radix {
                            (0..0x1_0000 / 4).collect()
                        } else {
                            src.clone()
                        },
                    ),
                    (2u8, if radix { firsts } else { dst.clone() }),
                    (3u8, vec![0u32; 0x1_0000 / 4]),
                    (4u8, vec![0u32; 0x1_0000 / 4]),
                    (5u8, vec![0u32; 0x1_0000 / 4]),
                ];
                // the radix sort's (slot 6): output stride 1, digit shift 0, 2048 keys
                let constants = [(0u8, constants), (6u8, vec![1, 0, 2048])];
                // `VICTORY_TEST_BVH_GROUPS` (4): groups of one dispatch. The radix sort's
                // inputs are one block's (its digits' first places, no block sums): more
                // groups would scatter over each other, in whatever order they run.
                let groups = if radix {
                    1
                } else {
                    std::env::var("VICTORY_TEST_BVH_GROUPS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(4)
                };
                let digest = |r: &Result<Vec<Vec<u32>>, String>| -> Result<u64, String> {
                    match r {
                        Ok(out) => Ok(out
                            .iter()
                            .flatten()
                            .fold(0xcbf2_9ce4_8422_2325u64, |h, &w| {
                                (h ^ u64::from(w)).wrapping_mul(0x100_0000_01b3)
                            })),
                        Err(e) => Err(e.clone()),
                    }
                };
                if !form.is_empty() {
                    // `VICTORY_TEST_BVH_REPEAT` (1): runs of the same inputs, which a
                    // deterministic program answers alike
                    let repeat: usize = std::env::var("VICTORY_TEST_BVH_REPEAT")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1);
                    // the radix sort's scatter (six root UAVs): keys and values sorted by
                    // the keys' low byte, stably, from the first places given
                    let sorted: Option<(Vec<u32>, Vec<u32>)> =
                        (picked.root_uavs.len() >= 6).then(|| {
                            let mut order: Vec<u32> = (0..2048).collect();
                            order.sort_by_key(|&i| keys[i as usize] & 0xFF);
                            (order.iter().map(|&i| keys[i as usize]).collect(), order)
                        });
                    for f in form.split(',') {
                        let shader = if f == "paired" { &paired } else { &picked };
                        let digests: Vec<String> = (0..repeat)
                            .map(|_| {
                                let r = run(&host, shader, &constants, &views, groups);
                                if let (Ok(out), Some((k, v))) = (&r, &sorted) {
                                    // root UAVs in slot order: keys out is slot 4, values slot 5
                                    let wrong = out[4][..2048]
                                        .iter()
                                        .zip(k)
                                        .filter(|(a, b)| a != b)
                                        .count()
                                        + out[5][..2048]
                                            .iter()
                                            .zip(v)
                                            .filter(|(a, b)| a != b)
                                            .count();
                                    eprintln!(
                                        "  {f}: {wrong} of 4096 dwords unlike the reference sort"
                                    );
                                }
                                match digest(&r) {
                                    Ok(d) => format!("{d:016x}"),
                                    Err(e) => e,
                                }
                            })
                            .collect();
                        let mut kinds = digests.clone();
                        kinds.sort();
                        kinds.dedup();
                        eprintln!(
                            "container {k} ({threads} threads) seed {seed} {f}: {} different of {repeat}{}",
                            kinds.len(),
                            if kinds.len() > 1 {
                                " NONDETERMINISTIC"
                            } else {
                                ""
                            }
                        );
                    }
                    continue;
                }
                let a = run(&host, &picked, &constants, &views, groups);
                let b = run(&host, &paired, &constants, &views, groups);
                match (a, b) {
                    (Ok(a), Ok(b)) => {
                        compared += 1;
                        if a != b {
                            let first = a.iter().zip(&b).enumerate().find_map(|(v, (x, y))| {
                                x.iter().zip(y).position(|(p, q)| p != q).map(|i| (v, i))
                            });
                            let n: usize = a
                                .iter()
                                .zip(&b)
                                .map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p != q).count())
                                .sum();
                            eprintln!(
                                "container {k} ({threads} threads) seed {seed}: {n} dwords differ, first {first:?}"
                            );
                            eprintln!(
                                "  src[0..8] {:x?} src[32..40] {:x?}",
                                &src[0..8],
                                &src[32..40]
                            );
                            for (v, (x, y)) in a.iter().zip(&b).enumerate() {
                                let at: Vec<String> = x
                                    .iter()
                                    .zip(y)
                                    .enumerate()
                                    .filter(|(_, (p, q))| p != q)
                                    .take(80)
                                    .map(|(i, (p, q))| format!("{i}:{p:#x}/{q:#x}"))
                                    .collect();
                                if !at.is_empty() {
                                    eprintln!("  view {v}: {}", at.join(" "));
                                }
                            }
                            differ.push((k, seed, n));
                        }
                    }
                    (a, b) => eprintln!("container {k}: not run ({:?} / {:?})", a.err(), b.err()),
                }
            }
        }
    }
    eprintln!("{compared} runs compared");
    assert!(differ.is_empty(), "the forms differ: {differ:?}");
}
