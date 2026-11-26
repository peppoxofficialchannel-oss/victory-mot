//! `VICTORY_ABLATE`: skip one stage of the pipeline at a time, to measure
//! what the most that stage can be worth (the image breaks; that is the
//! point of the measurement, not a bug).
//!
//! The value is a comma-separated list of stage names:
//!
//! * `heap` — the pass over the heap's buffer views (`draw::State::heap_buffers`);
//! * `copy` — the title-memory copies and refreshes prepared for a submission
//!   (`host::HostList::run_deferred`, `host::Batch::flush`);
//! * `copyback` — what shaders wrote copied back into title memory
//!   (`host::hand_back`, `host::relay_hand_back`);
//! * `sync` — the mirroring of descriptor records the title copied into its
//!   heaps (`descriptor::sync_heaps`);
//! * `textures` — the staging refresh of textures picked by index, inside a
//!   submission's preparation (`host::HostList::run_deferred`);
//! * `record` — the recording of the title's list calls into our host lists
//!   (`list::with_state`, except a list's reset).
//!
//! With the variable absent every stage runs exactly as it does today: each
//! call site is `if !ablated(BIT) { ... }`, and [`ablated`] is one relaxed
//! load of a mask parsed once ([`OnceLock`]). No allocation, no clock.

use std::sync::OnceLock;

/// The pass over the heap's buffer views.
pub const HEAP: u32 = 1 << 0;
/// The title-memory copies and refreshes prepared for a submission.
pub const COPY: u32 = 1 << 1;
/// What shaders wrote copied back into title memory.
pub const COPYBACK: u32 = 1 << 2;
/// The mirroring of descriptor records copied into shader-visible heaps.
pub const SYNC: u32 = 1 << 3;
/// The staging refresh of textures picked by index.
pub const TEXTURES: u32 = 1 << 4;
/// The recording of the title's list calls into our host lists.
pub const RECORD: u32 = 1 << 5;

/// Every stage, for the tests.
#[cfg(test)]
const ALL: u32 = HEAP | COPY | COPYBACK | SYNC | TEXTURES | RECORD;

/// The mask `VICTORY_ABLATE` names. Names not known are ignored; an empty
/// value is no ablation at all.
fn parse(names: &str) -> u32 {
    let mut mask = 0;
    for name in names.split(',') {
        mask |= match name.trim() {
            "heap" => HEAP,
            "copy" => COPY,
            "copyback" => COPYBACK,
            "sync" => SYNC,
            "textures" => TEXTURES,
            "record" => RECORD,
            _ => 0,
        };
    }
    mask
}

/// The stages this run skips, parsed once from `VICTORY_ABLATE`.
fn flags() -> u32 {
    static FLAGS: OnceLock<u32> = OnceLock::new();
    *FLAGS.get_or_init(|| {
        let mask = std::env::var("VICTORY_ABLATE").map_or(0, |v| parse(&v));
        if mask != 0 {
            let mut names: Vec<&str> = Vec::new();
            for (bit, name) in [
                (HEAP, "heap"),
                (COPY, "copy"),
                (COPYBACK, "copyback"),
                (SYNC, "sync"),
                (TEXTURES, "textures"),
                (RECORD, "record"),
            ] {
                if mask & bit != 0 {
                    names.push(name);
                }
            }
            tracing::info!(
                stages = %names.join(","),
                "stages skipped for the measurement (VICTORY_ABLATE): the image is expected to break"
            );
        }
        mask
    })
}

/// Whether the stage `bit` is skipped this run.
#[inline]
pub fn ablated(bit: u32) -> bool {
    flags() & bit != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ablate_names_parse_to_a_set() {
        assert_eq!(parse(""), 0);
        assert_eq!(parse("none"), 0);
        assert_eq!(parse("heap"), HEAP);
        assert_eq!(parse("heap,copy,copyback,sync,textures,record"), ALL);
        // Spaces, an empty entry and a name we do not know are ignored.
        assert_eq!(parse(" sync , copy ,, bogus"), SYNC | COPY);
        assert_eq!(parse("record,record"), RECORD);
    }
}
