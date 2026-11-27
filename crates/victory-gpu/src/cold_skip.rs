// Provenance: taken almost verbatim from `fxceless/victory-fork`, pull request
// #1358 ("VICTORY_COLD_SKIP=1|alt: refreshes pass over 2 MiB chunks with no
// write since"), file `crates/victory-gpu/src/cold_skip.rs`, +580/-72. That
// upstream (victory) is GPL-3.0; this file is under the same licence.
//
//! Refreshes of host copies that pass over the title memory nothing was
//! written in (`VICTORY_COLD_SKIP=1|alt`; off by default; the switch's two
//! halves are taken in turn in `crate::shadow`).
//!
//! A refresh walks every 64 KiB page of its range and compares the asking the
//! copy last took the page at with the asking that last found it written
//! (`crate::shadow::Shadow::copy_stale`). A large copy that many submissions
//! refresh is walked whole each time, mostly to find nothing.
//!
//! With the switch the journal of writes ([`Written`], `crate::watch`) numbers
//! every page it notes and keeps, per 2 MiB chunk, the number of the last
//! one: the chunk's mark. A copy keeps, per chunk, the mark it saw when a
//! walk of the whole chunk left every page [`fresh`] ([`Chunks::walked`]). A
//! fresh page becomes stale only by a new note, which moves the mark. So
//! while the mark has not moved no page of the chunk is stale, and the chunk
//! is passed over ([`Chunks::unchanged`]). The pages and the marks are read
//! under one lock, so a write noted after a walk's reading moves the mark
//! past the one the copy keeps.
//!
//! A page that is still stale after a walk took it (only part of it was in the
//! range, or its last write is numbered after the asking that stands) keeps
//! its chunk walked, so it is taken again at the next refresh as before.
//!
//! Standard library only, so that its tests run on any host.

use std::collections::BTreeMap;
use std::ops::Range;

/// Pages of a chunk: 2 MiB of 64 KiB pages.
pub(crate) const CHUNK_PAGES: usize = 32;

/// `VICTORY_COLD_SKIP`: 0 off (as without it), 1 on, 2 in turn (`alt`).
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_COLD_SKIP").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    })
}

/// Whether a page taken at asking `taken` (0: never) holds its last write,
/// found at asking `write` (0: none).
pub(crate) fn fresh(taken: u64, write: u64) -> bool {
    taken != 0 && write <= taken
}

/// The journal's pages: page number to the asking that last found it written,
/// and, with the switch on, each chunk's mark. Read as the `BTreeMap` it
/// holds; written by [`Written::insert`] alone, so that no write goes
/// unmarked. Marks are kept in both halves of `alt`.
pub(crate) struct Written {
    pages: BTreeMap<usize, u64>,
    marks: BTreeMap<usize, u64>,
    /// Numbers the notes.
    notes: u64,
}

impl Written {
    pub(crate) const fn new() -> Self {
        Written {
            pages: BTreeMap::new(),
            marks: BTreeMap::new(),
            notes: 0,
        }
    }

    /// Notes `page` found written at `asking`; the asking noted before.
    pub(crate) fn insert(&mut self, page: usize, asking: u64) -> Option<u64> {
        self.note(page, asking, mode() != 0)
    }

    fn note(&mut self, page: usize, asking: u64, mark: bool) -> Option<u64> {
        if mark {
            self.notes += 1;
            self.marks.insert(page / CHUNK_PAGES, self.notes);
        }
        self.pages.insert(page, asking)
    }

    /// The entries of pages `[first, last)`, and into `marks`, if asked for,
    /// the marks of the chunks that hold them, as (chunk, mark) in order.
    pub(crate) fn noted(
        &self,
        first: usize,
        last: usize,
        marks: Option<&mut Vec<(usize, u64)>>,
    ) -> Vec<(usize, u64)> {
        if let Some(marks) = marks {
            let chunks = first / CHUNK_PAGES..last.div_ceil(CHUNK_PAGES);
            marks.extend(self.marks.range(chunks).map(|(&c, &m)| (c, m)));
        }
        self.pages
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect()
    }

    /// For tests: forgets `page`, as if it had never been noted. Its chunk's
    /// mark stays: marks only move forward.
    #[cfg(test)]
    pub(crate) fn forget(&mut self, page: usize) {
        self.pages.remove(&page);
    }
}

impl std::ops::Deref for Written {
    type Target = BTreeMap<usize, u64>;

    fn deref(&self) -> &Self::Target {
        &self.pages
    }
}

/// Pages `[first, last]` cut where chunks begin.
pub(crate) fn spans(first: usize, last: usize) -> impl Iterator<Item = Range<usize>> {
    (first / CHUNK_PAGES..=last / CHUNK_PAGES).map(move |chunk| {
        (chunk * CHUNK_PAGES).max(first)..((chunk + 1) * CHUNK_PAGES).min(last + 1)
    })
}

/// What a copy knows of its chunks, for one walk.
pub(crate) struct Chunks<'a> {
    /// The copy's pages.
    pages: Range<usize>,
    /// Per chunk from the one that holds the copy's first page: the mark seen
    /// when a walk last left the chunk fresh, plus one (0: never).
    verified: &'a mut Vec<u64>,
    /// The marks read with the walk's entries ([`Written::noted`]).
    marks: &'a [(usize, u64)],
}

impl<'a> Chunks<'a> {
    /// `verified` is the copy's own, empty before its first walk.
    pub(crate) fn new(
        verified: &'a mut Vec<u64>,
        pages: Range<usize>,
        marks: &'a [(usize, u64)],
    ) -> Self {
        verified.resize(
            pages.end.div_ceil(CHUNK_PAGES) - pages.start / CHUNK_PAGES,
            0,
        );
        Chunks {
            pages,
            verified,
            marks,
        }
    }

    /// Where `verified` keeps the chunk of `page`, and the chunk's mark.
    fn of(&self, page: usize) -> (usize, u64) {
        let chunk = page / CHUNK_PAGES;
        let mark = self
            .marks
            .binary_search_by_key(&chunk, |m| m.0)
            .map_or(0, |i| self.marks[i].1);
        (chunk - self.pages.start / CHUNK_PAGES, mark)
    }

    /// Whether no page of `span` (one of [`spans`]) can be stale.
    pub(crate) fn unchanged(&self, span: &Range<usize>) -> bool {
        let (at, mark) = self.of(span.start);
        self.verified[at] != 0 && mark < self.verified[at]
    }

    /// `span` was walked, and `all_fresh` says that every page of it is fresh
    /// now: remembered if it held all of the copy's pages in its chunk.
    pub(crate) fn walked(&mut self, span: &Range<usize>, all_fresh: bool) {
        let chunk = span.start / CHUNK_PAGES;
        let whole = span.start <= (chunk * CHUNK_PAGES).max(self.pages.start)
            && span.end >= ((chunk + 1) * CHUNK_PAGES).min(self.pages.end);
        if all_fresh && whole {
            let (at, mark) = self.of(span.start);
            self.verified[at] = mark + 1;
        }
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)] // a walk's runs of one are meant
mod tests {
    use super::*;

    /// A copy and a journal, walked as `Shadow::copy_stale` walks them.
    struct Rig {
        pages: Range<usize>,
        copied: Vec<u64>,
        verified: Vec<u64>,
        written: Written,
        asked: u64,
    }

    /// The runs of pages a refresh copied, and the pages it passed over.
    type Walk = (Vec<Range<usize>>, usize);

    impl Rig {
        fn new(base: usize, pages: usize) -> Self {
            Rig {
                pages: base..base + pages,
                copied: vec![0; pages],
                verified: Vec::new(),
                written: Written::new(),
                asked: 0,
            }
        }

        fn ask(&mut self) -> u64 {
            self.asked += 1;
            self.asked
        }

        fn write(&mut self, page: usize) {
            let asking = self.ask();
            self.written.note(page, asking, true);
        }

        fn refresh(&mut self, first: usize, last: usize, skip: bool) -> Walk {
            let now = self.ask();
            self.refresh_at(now, first, last, skip, |_| true)
        }

        /// A refresh of pages `[first, last]` at asking `now`; `whole` says
        /// whether the range holds all of a page.
        fn refresh_at(
            &mut self,
            now: u64,
            first: usize,
            last: usize,
            skip: bool,
            whole: impl Fn(usize) -> bool,
        ) -> Walk {
            let mut marks = Vec::new();
            let written: BTreeMap<_, _> = self
                .written
                .noted(first, last + 1, skip.then_some(&mut marks))
                .into_iter()
                .collect();
            let mut chunks =
                skip.then(|| Chunks::new(&mut self.verified, self.pages.clone(), &marks));
            let (mut runs, mut run, mut passed) = (Vec::new(), None, 0);
            for span in spans(first, last) {
                if chunks.as_ref().is_some_and(|c| c.unchanged(&span)) {
                    runs.extend(run.take().map(|a| a..span.start));
                    passed += span.len();
                    continue;
                }
                let mut all_fresh = true;
                for page in span.clone() {
                    let write = written.get(&page).copied().unwrap_or(0);
                    let at = &mut self.copied[page - self.pages.start];
                    if !fresh(*at, write) {
                        if whole(page) {
                            *at = now;
                        }
                        all_fresh &= fresh(*at, write);
                        run.get_or_insert(page);
                    } else {
                        runs.extend(run.take().map(|a| a..page));
                    }
                }
                if let Some(chunks) = chunks.as_mut() {
                    chunks.walked(&span, all_fresh);
                }
            }
            runs.extend(run.map(|a| a..last + 1));
            (runs, passed)
        }
    }

    #[test]
    fn spans_are_cut_where_chunks_begin() {
        assert_eq!(spans(5, 5).collect::<Vec<_>>(), [5..6]);
        assert_eq!(spans(31, 64).collect::<Vec<_>>(), [31..32, 32..64, 64..65]);
    }

    #[test]
    fn the_first_walk_takes_all_and_the_next_passes_over_all() {
        let mut r = Rig::new(40, 100);
        assert_eq!(r.refresh(40, 139, true), (vec![40..140], 0));
        assert_eq!(r.refresh(40, 139, true), (vec![], 100));
    }

    #[test]
    fn a_write_has_its_chunk_walked_and_no_other() {
        let mut r = Rig::new(0, 128);
        r.refresh(0, 127, true);
        r.write(70);
        assert_eq!(r.refresh(0, 127, true), (vec![70..71], 96));
        assert_eq!(r.refresh(0, 127, true), (vec![], 128));
    }

    #[test]
    fn a_chunk_walked_in_part_is_not_remembered() {
        let mut r = Rig::new(0, 64);
        r.refresh(0, 39, true);
        assert_eq!(r.refresh(0, 63, true), (vec![40..64], 32));
    }

    #[test]
    fn a_page_taken_in_part_keeps_its_chunk_walked() {
        let mut r = Rig::new(0, 32);
        let now = r.ask();
        r.refresh_at(now, 0, 31, true, |p| p != 5);
        assert_eq!(r.refresh(0, 31, true), (vec![5..6], 0));
    }

    #[test]
    fn a_write_numbered_after_the_standing_asking_keeps_its_chunk_walked() {
        let mut r = Rig::new(0, 32);
        r.refresh(0, 31, true);
        let standing = r.ask();
        r.write(3);
        // Without the switch the page is taken at every refresh until a later
        // asking: so it is with it.
        for _ in 0..2 {
            assert_eq!(
                r.refresh_at(standing, 0, 31, true, |_| true),
                (vec![3..4], 0)
            );
        }
        assert_eq!(r.refresh(0, 31, true), (vec![3..4], 0));
        assert_eq!(r.refresh(0, 31, true), (vec![], 32));
    }

    #[test]
    fn a_passed_over_chunk_ends_a_run_where_it_starts() {
        let mut r = Rig::new(0, 96);
        r.refresh(32, 63, true);
        assert_eq!(r.refresh(0, 95, true), (vec![0..32, 64..96], 32));
    }

    #[test]
    fn the_same_runs_with_and_without_the_skip() {
        let (mut on, mut off) = (Rig::new(17, 300), Rig::new(17, 300));
        let mut x: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = move |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n as u64) as usize
        };
        let (mut standing, mut passed) = (None, 0);
        for _ in 0..20_000 {
            match next(8) {
                0 | 1 => {
                    let p = 17 + next(300);
                    on.write(p);
                    off.write(p);
                }
                // An asking that stands for the refreshes that follow.
                2 => {
                    let asking = on.ask();
                    assert_eq!(asking, off.ask());
                    standing = Some(asking);
                }
                3 => standing = None,
                _ => {
                    let a = 17 + next(300);
                    let b = (a + next(200)).min(316);
                    let new = on.ask();
                    assert_eq!(new, off.ask());
                    let now = standing.unwrap_or(new);
                    // The range may hold its end pages only in part.
                    let (head, tail) = (next(2) == 0, next(2) == 0);
                    let whole = move |p| !(p == a && head || p == b && tail);
                    let (runs_on, skipped) = on.refresh_at(now, a, b, true, whole);
                    let (runs_off, none) = off.refresh_at(now, a, b, false, whole);
                    assert_eq!(runs_on, runs_off);
                    assert_eq!(on.copied, off.copied);
                    assert_eq!(none, 0);
                    passed += skipped;
                }
            }
        }
        assert!(passed > 0);
    }

    #[test]
    fn every_note_moves_its_chunks_mark_and_none_while_off() {
        let mut w = Written::new();
        w.note(3, 9, false);
        let mut marks = Vec::new();
        assert_eq!(w.noted(0, 64, Some(&mut marks)), [(3, 9)]);
        assert!(marks.is_empty());
        w.note(3, 9, true);
        w.note(33, 9, true);
        w.noted(0, 64, Some(&mut marks));
        assert_eq!(marks, [(0, 1), (1, 2)]);
        // The same page with an asking no newer moves its mark still.
        w.note(3, 9, true);
        marks.clear();
        w.noted(0, 32, Some(&mut marks));
        assert_eq!(marks, [(0, 3)]);
        assert_eq!(w.get(&33), Some(&9), "read as the map it holds");
    }
}
