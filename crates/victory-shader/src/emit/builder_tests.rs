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

/// A compute program of `threads` threads a group (a power of two; `wave64`: in console waves
/// of 64 lanes) that stores, at dword `group * threads + thread` of its root UAV (root
/// parameter 0, user data 0 and 1, whose descriptor it builds as the builder's programs do),
/// the thread's index in its group plus 1. The group id arrives in s2, after the address.
fn numbering(wave64: bool, threads: u32) -> NativeProgram {
    use crate::native::{ComputeIo, ResourceType, RootParam, UsedResource};
    let words: Vec<u32> = vec![
        0x8F0C_0002 | (0x80 + threads.trailing_zeros()) << 8, // s_lshl_b32 s12, s2, log2(threads)
        0x8801_FF01,
        0x0001_0000, // s_or_b32 s1, s1, 0x10000 (a stride of 1)
        0xBE82_03C1, // s_mov_b32 s2, -1 (records)
        0xBE83_03FF,
        0x1101_4FAC, // s_mov_b32 s3, 0x11014fac
        0x4A02_000C, // v_add_nc_u32 v1, s12, v0
        0x3404_0282, // v_lshlrev_b32 v2, 2, v1
        0x4A06_0081, // v_add_nc_u32 v3, 1, v0
        0xE070_1000,
        0x8000_0302, // buffer_store_dword v3, v2, s[0:3], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 1,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        // two user SGPRs (the address), then the group id x
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1 | 1 << 7,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// A console wave64 program of 64 threads a group that reads its wave across the halves: each
/// thread stores, at dwords `2 (group * 64 + thread)` and the next of its root UAV, its rank
/// among the lanes in EXEC (`v_mbcnt`) and their count (`s_bcnt1_i32_b64` of EXEC). On a host
/// of 32-lane waves the two host waves of a console wave exchange both.
fn ranking() -> NativeProgram {
    let mut native = numbering(true, 64);
    let words: Vec<u32> = vec![
        0x8F0C_8602, // s_lshl_b32 s12, s2, 6
        0x8801_FF01,
        0x0001_0000, // s_or_b32 s1, s1, 0x10000
        0xBE82_03C1, // s_mov_b32 s2, -1
        0xBE83_03FF,
        0x1101_4FAC, // s_mov_b32 s3, 0x11014fac
        0xBE84_107E, // s_bcnt1_i32_b64 s4, exec
        0xD765_0001,
        0x0001_007E, // v_mbcnt_lo_u32_b32 v1, exec_lo, 0
        0xD766_0001,
        0x0002_027F, // v_mbcnt_hi_u32_b32 v1, exec_hi, v1
        0x4A04_000C, // v_add_nc_u32 v2, s12, v0
        0x3404_0483, // v_lshlrev_b32 v2, 3, v2
        0x7E06_0204, // v_mov_b32 v3, s4
        0xE070_1000,
        0x8000_0102, // buffer_store_dword v1, v2, s[0:3], 0 offen
        0xE070_1004,
        0x8000_0302, // buffer_store_dword v3, v2, s[0:3], 0 offen offset:4
        0xBF81_0000, // s_endpgm
    ];
    native.code = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    native
}

/// A console wave whose lanes past a dispatch's thread count were never launched counts only
/// the launched ones: its EXEC, and each lane's rank in it. On a host of 32-lane waves the
/// console wave is two host waves that exchange their halves, and in a group cut short at 32
/// lanes or fewer the upper one has no launched thread at all: it gives "no lanes" to every
/// exchange and leaves. Run on the host GPU when it takes waves of 32.
#[test]
fn a_wave_cut_short_by_a_thread_count_counts_its_launched_lanes() {
    let native = ranking();
    let limited = lowered_for(&native, true).expect("lowered for thread counts");
    if let Ok(mut v) = super::validator::Validator::load(super::validator::Validator::default_path().as_deref()) {
        let mut copy = limited.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline_bounded(&limited) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    for threads in [6u32, 32, 45, 64 + 33, 128] {
        let mut ud = vec![0u32; limited.user_data_dwords().max(1) as usize];
        for u in &limited.user_data {
            if u.root == RootType::Other(super::USER_THREAD_LIMIT) {
                ud[u.index as usize] = [threads, 1, 1][u.dword.min(2) as usize];
            }
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud)), (1, Root::Uav(vec![0; 512]))],
                    views: &[],
                    samplers: &[],
                    groups: [threads.div_ceil(64), 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch: {e}"));
        let out = out.into_iter().last().unwrap_or_default();
        let want: Vec<u32> = (0..256u32)
            .flat_map(|i| {
                if i < threads {
                    let group = i / 64;
                    let launched = (threads - group * 64).min(64);
                    [i % 64, launched]
                } else {
                    [0, 0]
                }
            })
            .collect();
        assert_eq!(out[..512], want[..], "{threads} threads");
    }
    eprintln!("ran: the cut-short waves counted their launched lanes");
}

/// `native` lowered for a host of 32-lane waves; `limited`: for dispatches by thread counts
/// ([`lower::with_thread_limits`]).
fn lowered_for(native: &NativeProgram, limited: bool) -> Result<HostShader, crate::ShaderError> {
    let p = crate::spill::build(native)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, native)?;
    let parts = lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    };
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let lowered = if limited {
        lower::with_thread_limits(|| lower::lower(&parts))
    } else {
        lower::lower(&parts)
    };
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers.iter().map(|&(w, _)| w).collect(),
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    })
}

/// The console's `DispatchX` with `USE_THREAD_DIMENSIONS` counts threads: the last group is
/// cut short, and no thread past the count runs (the builder rebuilds the levels of its
/// pre-built trees in place, one node per thread, and a thread past a level's nodes would
/// write a node over the next level's input). The host runs whole groups of the program in
/// its form for that, given the count: every thread inside it stores, none past it does. In
/// the plain form the whole last group stores. Run on the host GPU (or on WARP with
/// `VICTORY_WARP=1`: nothing here asks for a wave size).
#[test]
fn a_dispatch_by_thread_counts_runs_no_thread_past_them() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut validator =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref()).ok();
    let mut shaders = Vec::new();
    for (wave64, group) in [(false, 32u32), (true, 64)] {
        let native = numbering(wave64, group);
        let limited = lowered_for(&native, true).expect("lowered for thread counts");
        let plain = lowered_for(&native, false).expect("lowered");
        assert!(
            limited
                .user_data
                .iter()
                .any(|u| u.root == RootType::Other(super::USER_THREAD_LIMIT)),
            "the counts are read"
        );
        if let Some(v) = validator.as_mut() {
            for s in [&limited, &plain] {
                let mut copy = s.dxil.clone();
                v.validate(&mut copy).expect("DXIL validation failed");
            }
        }
        shaders.push((group, limited, plain));
    }
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 device; not run");
        return;
    };
    for (group, limited, plain) in &shaders {
        let group = *group;
        for threads in [group / 4 + 1, group + group / 2 - 3, 2 * group] {
            let groups = threads.div_ceil(group);
            let run = |shader: &HostShader| -> Vec<u32> {
                let (root, pso) = host
                    .compute_pipeline_bounded(shader)
                    .unwrap_or_else(|e| panic!("pipeline: {e}"));
                let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
                for u in &shader.user_data {
                    if u.root == RootType::Other(super::USER_THREAD_LIMIT) {
                        ud[u.index as usize] = [threads, 1, 1][u.dword.min(2) as usize];
                    }
                }
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(ud)), (1, Root::Uav(vec![0; 256]))],
                            views: &[],
                            samplers: &[],
                            groups: [groups, 1, 1],
                        },
                    )
                    .unwrap_or_else(|e| panic!("dispatch: {e}"));
                out.into_iter().last().unwrap_or_default()
            };
            let want: Vec<u32> = (0..256).map(|i| if i < threads { i % group + 1 } else { 0 }).collect();
            let out = run(limited);
            assert_eq!(out[..256], want[..], "{threads} threads in groups of {group}");
            let whole = run(plain);
            let n = threads as usize;
            assert_eq!(whole[..n], want[..n], "the plain form, inside the count");
            if !threads.is_multiple_of(group) {
                assert_ne!(whole[n], 0, "the plain form runs the whole last group");
            }
        }
    }
    eprintln!("ran: no thread past the counts stored");
}

/// A group barrier must be met by the whole group, and a group cut short has threads the host
/// never launches: in D3D12 the barrier those never reach is undefined, so a program that meets
/// one is refused for dispatches by thread counts ([`lower::LIMITS_BARRIER`]) and still
/// translates for group counts. Translates only; nothing runs.
#[test]
fn a_dispatch_by_thread_counts_refuses_a_group_barrier() {
    let plain = numbering(false, 32);
    // the same program, the group meeting a barrier before it stops
    let mut words: Vec<u32> = plain
        .code
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    let end = words.iter().position(|&w| w == 0xBF81_0000).expect("s_endpgm");
    words.insert(end, 0xBF8A_0000); // s_barrier
    let barrier = NativeProgram {
        code: words.into_iter().flat_map(|w| w.to_le_bytes()).collect(),
        ..plain
    };
    let error = lowered_for(&barrier, true).expect_err("a cut-short group meeting a barrier");
    assert!(
        matches!(
            error,
            crate::ShaderError::Unsupported { what, .. } if what == lower::LIMITS_BARRIER
        ),
        "{error}"
    );
    lowered_for(&barrier, false).expect("group counts meet the barrier whole");
}

/// The builder's programs translated for dispatches by thread counts
/// ([`super::translate_thread_limited`]; the title dispatches its box rehydration, header
/// copies and triangle reloads so): each translates as it does for group counts, but in the
/// paired and the approximate forms, which are refused (a thread holding two console lanes
/// cannot leave for one of them), and validates. `VICTORY_TEST_BVH_SAVE=<dir>`: each
/// translation's DXIL container there (`<n>.threads.dxil`, n the container's place in the
/// file), for a look with a disassembler. Translates only; nothing runs.
#[test]
fn the_builders_programs_translate_for_dispatches_by_thread_counts() {
    let Some(path) = std::env::var_os("VICTORY_TEST_BVH_DLL") else {
        eprintln!("VICTORY_TEST_BVH_DLL not set; not run");
        return;
    };
    let bytes = std::fs::read(&path).expect("the builder DLL");
    let save = std::env::var_os("VICTORY_TEST_BVH_SAVE").map(std::path::PathBuf::from);
    let mut validator =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref()).ok();
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let (mut done, mut refused, mut wrong) = (0, 0, Vec::new());
    let mut barriers = 0;
    // the dwords of the root signature the layer makes (`pipeline/host.rs`): the user data as
    // root constants, root CBVs as root descriptors, the title's root SRVs and UAVs as
    // one-view tables, ours as root descriptors; D3D12 allows 64
    let ours = |s: u8| s >= super::SLOT_LOOPS;
    let cost = |h: &HostShader| {
        h.user_data_dwords()
            + 2 * h.root_cbvs.len() as u32
            + h.root_srvs.iter().chain(&h.root_uavs).map(|&s| if ours(s) { 2 } else { 1 }).sum::<u32>()
    };
    let mut most = (0, 0);
    for (k, container) in containers(&bytes).into_iter().enumerate() {
        let plain = super::translate(container);
        let limited = super::translate_thread_limited(container);
        match (plain, limited) {
            (Err(_), _) => {}
            (Ok(plain), Ok(shader)) => {
                done += 1;
                most = (most.0.max(cost(&plain)), most.1.max(cost(&shader)));
                if cost(&shader) > 64 {
                    wrong.push((k, format!("a root signature of {} dwords", cost(&shader))));
                }
                let mut dxil = shader.dxil.clone();
                if let Some(v) = validator.as_mut()
                    && let Err(e) = v.validate(&mut dxil)
                {
                    wrong.push((k, e));
                }
                if let Some(dir) = &save {
                    let _ = std::fs::create_dir_all(dir);
                    let _ = std::fs::write(dir.join(format!("{k}.threads.dxil")), &shader.dxil);
                }
            }
            (Ok(_), Err(crate::ShaderError::Unsupported { what, .. })) if what == lower::LIMITS_INEXACT => {
                refused += 1;
            }
            // (a cut-short group that meets a group barrier: D3D12 leaves it undefined, so it
            // is refused too — the count is what a run's diagnostics line reports)
            (Ok(_), Err(crate::ShaderError::Unsupported { what, .. })) if what == lower::LIMITS_BARRIER => {
                refused += 1;
                barriers += 1;
            }
            (Ok(_), Err(e)) => wrong.push((k, e.to_string())),
        }
    }
    lower::TEST_HOST_WAVE.set(was);
    eprintln!(
        "{done} translate for thread counts, {refused} refused ({barriers} of them for a group barrier, the rest paired or approximate forms); root signatures of {} dwords at most ({} for group counts)",
        most.1, most.0
    );
    assert!(wrong.is_empty(), "{wrong:?}");
}
