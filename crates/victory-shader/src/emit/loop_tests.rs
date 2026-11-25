//! End-to-end tests of translated loops: the iteration limit and the loop
//! notes (`shader::SLOT_LOOPS`). Synthetic GFX10 compute programs lowered to
//! DXIL, validated by `dxil.dll` when present, and run on the host GPU and
//! WARP.

use super::host::{Dispatch, Host, Root, View};
use super::lower::{COMPUTE_LOOP_LIMIT, TEST_NOTES};
use super::valu_tests::{ONE_AT_A_TIME, build_cs, s_mov_b32, sgpr, sop2, vop1};
use super::{HostShader, LOOP_NOTE_DWORDS, LOOP_NOTES, SLOT_LOOPS};
use crate::native::RootType;

/// `s_add_u32 s6, s6, 1`.
const S_ADD_S6_1: u32 = sop2(0x00, 6, 6, 0x81);
/// `s_cmp_lg_u32 s6, s0`.
const S_CMP_LG_S6_S0: u32 = 0xBF07_0006;
/// `s_cmp_eq_u32 s6, s0`.
const S_CMP_EQ_S6_S0: u32 = 0xBF06_0006;

/// What the 32 threads of `body` (with [`build_cs`]'s prologue of 7 dwords)
/// stored, and the program's loop notes, on each device that ran it, with
/// the root constants all 0.
fn run(body: &[u32]) -> Vec<(&'static str, u32, Vec<u32>)> {
    TEST_NOTES.set(true);
    let shader = build_cs(body, 32, 0);
    TEST_NOTES.set(false);
    assert_eq!(shader.root_uavs, [SLOT_LOOPS]);
    run_shader(&shader)
}

fn run_shader(shader: &HostShader) -> Vec<(&'static str, u32, Vec<u32>)> {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline_bounded(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        // The root constants and the table's heap index: all 0.
        for u in &shader.user_data {
            assert!(matches!((u.root, u.slot), (RootType::Constants, 0) | (RootType::Table, 1)), "{u:?}");
        }
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        // Root parameter 0 is the user data, 1 the notes (the only root view).
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (1, Root::Uav(vec![0; LOOP_NOTES as usize])),
                    ],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words[0][0], words[1].clone()));
    }
    assert!(!out.is_empty(), "no SM 6.6 device");
    out
}

/// The record of loop `k` in a program's notes.
fn record(notes: &[u32], k: usize) -> &[u32] {
    let n = LOOP_NOTE_DWORDS as usize;
    &notes[k * n..(k + 1) * n]
}

#[test]
fn a_loop_whose_branch_back_holds_leaves_at_the_limit_and_goes_on() {
    // s6 counts up from 0 while it differs from s0 (0): the console would
    // run it four billion times. The loop starts at byte 32.
    let body = [
        s_mov_b32(6, 0x80),
        S_ADD_S6_1,
        S_CMP_LG_S6_S0,
        0xBF85_FFFD, // s_cbranch_scc1 (back to the s_add)
        vop1(0x001, 0, sgpr(6)), // v_mov_b32 v0, s6
    ];
    for (label, stored, notes) in run(&body) {
        // The lanes went on after the last allowed pass and stored s6.
        assert_eq!(stored, COMPUTE_LOOP_LIMIT, "{label}");
        // Each of the 32 counted as leaving the loop and going on.
        assert_eq!(record(&notes, 0), [32 + 1, 32, 0, 0], "{label}");
        assert!(notes[LOOP_NOTE_DWORDS as usize..].iter().all(|&w| w == 0), "{label}");
    }
}

#[test]
fn a_loop_closed_by_a_jump_takes_the_lane_out_at_the_limit() {
    // The same count, left by a forward branch when s6 reaches s0 (0) and
    // closed by `s_branch`: at the limit the lane leaves the program.
    let body = [
        s_mov_b32(6, 0x80),
        S_ADD_S6_1,
        S_CMP_EQ_S6_S0,
        0xBF85_0001, // s_cbranch_scc1 (past the s_branch)
        0xBF82_FFFC, // s_branch (back to the s_add)
        vop1(0x001, 0, sgpr(6)), // v_mov_b32 v0, s6
    ];
    for (label, stored, notes) in run(&body) {
        // Nothing stored: every lane was taken out.
        assert_eq!(stored, 0, "{label}");
        assert_eq!(record(&notes, 0), [32 + 1, 0, 32, 0], "{label}");
    }
}

#[test]
fn a_loop_that_ends_leaves_no_note() {
    // s6 counts to 5 (s0 + 5 written as an inline constant compare).
    let body = [
        s_mov_b32(6, 0x80),
        S_ADD_S6_1,
        0xBF07_8506, // s_cmp_lg_u32 s6, 5
        0xBF85_FFFD, // s_cbranch_scc1
        vop1(0x001, 0, sgpr(6)),
    ];
    for (label, stored, notes) in run(&body) {
        assert_eq!(stored, 5, "{label}");
        assert!(notes.iter().all(|&w| w == 0), "{label}: {notes:?}");
    }
}
