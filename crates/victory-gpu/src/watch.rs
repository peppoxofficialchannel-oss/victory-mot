//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! Writes into physical pages through another view are seen too: the
//! runtime tells us the views of the pages it wrote itself ([`writing`],
//! xmem's `write_physical` and DirectStorage's page destinations), and a
//! write by the title through any view of a guarded page faults like one
//! at the guarded address (the guard makes every view of the page
//! read-only, [`open`] gives them all back; not a page of more than
//! [`MANY_VIEWS`] views). A
//! system call writing into a guarded block fails (none is known to).

use std::collections::BTreeMap;
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

// From cool-fox (GPL-3.0, commit f4b48ad): the mask of a page's parts, and
// cutting a refresh's runs to the parts written.
/// The parts of a page.
const PARTS: usize = PAGE / KERNEL_PAGE;
/// All of them.
const WHOLE: u16 = u16::MAX;

/// Whether the parts of a page that the kernel names are kept, and a refresh
/// copies those (`VICTORY_TAKE_PARTS=1`; absent or `0`: the 64 KiB page, as
/// before).
///
/// The kernel says which 4 KiB pages were written, and a copy took the 64 KiB
/// around each: a title writing 4 KiB in a 64 KiB page was copied sixteen
/// times over. Our logs: 172.7 MB copied a frame for the bytes written
/// (`faults = 634 x MB_copied - 11,354`, r = 0.904), 124-167 ms of a frame
/// (long21, long22). The donor measured 114 MB copied for 29 MB written
/// (session play-1125-0203, run cpu-2). What is checked is as before, a page.
pub(crate) fn parts_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TAKE_PARTS").is_some_and(|v| v == "1"))
}

/// A title-memory range whose changed pages another subsystem needs to read.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    seen: Mutex<BTreeMap<usize, u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if len == 0 || writes(start, end, views).is_none() {
        return None;
    }
    Some(Tracked {
        start,
        end,
        seen: Mutex::default(),
    })
}

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// The same for each of a page's sixteen parts ([`KERNEL_PAGE`]), where
    /// the parts are known ([`parts_on`]); a page that is not here was
    /// written whole, as far as anyone knows.
    parts: BTreeMap<usize, [u64; PARTS]>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Vec<(usize, usize)>,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    parts: BTreeMap::new(),
    found: Vec::new(),
    ahead: Vec::new(),
    ahead_asked: 0,
    kept: false,
});

/// The parts of the page numbered `page` that the bytes `[at, end)` touch (a
/// bit for each 4 KiB): the mask the kernel's answer would give for a write
/// of ours into a known range. `WHOLE` if the range is not in the page.
fn parts_in(page: usize, at: usize, end: usize) -> u16 {
    let (a, b) = ((page * PAGE).max(at), ((page + 1) * PAGE).min(end));
    if a >= b {
        return WHOLE;
    }
    let first = (a - page * PAGE) / KERNEL_PAGE;
    let past = (b - page * PAGE).div_ceil(KERNEL_PAGE);
    (u16::MAX << first) & (u16::MAX >> (PARTS - past))
}

impl Journal {
    /// Page `page` was found written by asking `now`: the parts in `parts`
    /// (a bit each). The asking that found it written before (0: none).
    fn note(&mut self, page: usize, parts: u16, now: u64) -> u64 {
        let before = self.written.insert(page, now).unwrap_or(0);
        if parts == WHOLE || !parts_on() {
            self.parts.remove(&page);
            return before;
        }
        let all = self.parts.entry(page).or_insert([before; PARTS]);
        for (k, at) in all.iter_mut().enumerate() {
            if parts >> k & 1 != 0 {
                *at = now;
            }
        }
        before
    }

    /// The parts of `page` written by an asking after `taken`.
    fn parts_since(&self, page: usize, taken: u64) -> u16 {
        match self.parts.get(&page) {
            Some(all) => all.iter().enumerate().filter(|(_, at)| **at > taken).fold(0, |m, (k, _)| m | 1 << k),
            None => WHOLE,
        }
    }
}

/// Of the pages of a copy that were written after it took them (page number,
/// the asking it took the page at): the parts to take again, a bit for each
/// 4 KiB, in the same order.
pub(crate) fn parts_written(pages: &[(usize, u64)]) -> Vec<u16> {
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let parts: Vec<u16> = pages.iter().map(|&(page, taken)| j.parts_since(page, taken)).collect();
    #[cfg(feature = "diagnostics")]
    count_parts(&parts);
    parts
}

/// Since the start: pages a refresh took again with a real mask, and whole
/// (a line every 65,536 calls; `VICTORY_TAKE_PARTS=1` only).
#[cfg(feature = "diagnostics")]
fn count_parts(parts: &[u16]) {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static SEEN: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
    if !parts_on() || parts.is_empty() {
        return;
    }
    let masked = parts.iter().filter(|&&p| p != WHOLE).count() as u64;
    SEEN[1].fetch_add(masked, Relaxed);
    SEEN[2].fetch_add(parts.len() as u64 - masked, Relaxed);
    let calls = SEEN[0].fetch_add(1, Relaxed) + 1;
    if calls.is_multiple_of(65536) {
        tracing::info!(
            calls,
            masked = SEEN[1].load(Relaxed),
            whole = SEEN[2].load(Relaxed),
            "pages taken again with the 4 KiB parts written, and whole (VICTORY_TAKE_PARTS=1), since the start"
        );
    }
}

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`.
    if j.kept {
        return;
    }
    // View memory the guards hold is left to `writes`, which answers it
    // from the guards ([`known_view`]): the kernel's write watch can only
    // fail there. Asked here with the others, a run holding such a range
    // failed, was asked again by halves down to single ranges, failing
    // each time, and the range then went to `writes` all the same (after
    // triippywhale d0581a55). `VICTORY_VIEW_ASK=1` asks as before.
    #[cfg(feature = "diagnostics")]
    let all = pieces.len();
    pieces.retain(|&(first, _)| !known_view(first * PAGE));
    #[cfg(feature = "diagnostics")]
    AHEAD_VIEWS.fetch_add((all - pieces.len()) as u64, std::sync::atomic::Ordering::Relaxed);
    j.asked += 1;
    j.ahead_asked = j.asked;
    j.ahead.clear();
    // The kernel's answer names the 4 KiB page that was written, not the 64
    // KiB one: `ask_kernel` notes which part it is (`VICTORY_TAKE_PARTS`),
    // so the copies that `writes` answers from here can be cut to it.
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + AHEAD_GAP
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        ask_run(j, &pieces[i..k]);
        i = k;
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// Since the last frame line: ranges of view memory the guards hold that a
/// submission's askings ahead ([`ask_ahead`], [`ask_once`]) left out. Each
/// was a kernel asking that failed, or several, before.
#[cfg(feature = "diagnostics")]
pub(crate) static AHEAD_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    ask_once_end();
    ask_ahead(ranges);
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    if j.ahead.is_empty() {
        // `ask_ahead` asked nothing (fewer than two ranges).
        j.asked += 1;
        j.ahead_asked = j.asked;
    }
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    for &(start, end) in ranges {
        if start >= end {
            continue;
        }
        // View memory the guards hold is not asked about: the asking can
        // only fail there, one failed kernel call for each such copy at
        // every submission, and `writes` answers it from the guards
        // (after triippywhale cc368c7c).
        if known_view(start) {
            #[cfg(feature = "diagnostics")]
            AHEAD_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if !j.ahead.iter().any(|&(a, b)| a <= first && last <= b) && ask_kernel(j, first, last) {
            j.ahead.push((first, last));
        }
    }
    j.kept = !j.ahead.is_empty();
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept || len == 0 {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    // The asking that stands moves with it: `asked` and `writes` give it as
    // "now", and a page noted written after "now" stays stale for every
    // copy that takes it (copied at `now`, written at `now + 1`): copied
    // again at every refresh, and a reason to settle flights
    // (fxceless/victory-fork#410).
    j.ahead_asked = now;
    // The bytes are ours and known: the parts they touch are the mask the
    // kernel would give (not `WHOLE`, which would throw away the offset).
    let end = at.saturating_add(len);
    for page in at / PAGE..end.div_ceil(PAGE) {
        j.note(page, parts_in(page, at, end), now);
    }
}

/// We wrote `[at, at + len)` of title memory through a view of our own
/// (`crate::inplace::write_own`): no guard was opened and no kernel
/// write-watch bit set, so the journal is told here. The page holding `at`,
/// and the same physical page at its other title addresses (`others`, page
/// numbers), count as written at a new asking, which every host copy that
/// holds them takes at its next refresh. Returns the asking that had last
/// found the page written (0: never) and the new one, for the copy that
/// wrote to note itself current.
pub(crate) fn wrote_own(at: usize, len: usize, others: &[usize]) -> (u64, u64) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    j.asked += 1;
    let now = j.asked;
    // As in [`wrote`]: an asking that stands for a submission, or for a
    // flush, moves with it, or a copy taking the page at the standing
    // asking would find it written "later" at every refresh.
    if j.kept || !j.ahead.is_empty() {
        j.ahead_asked = now;
    }
    // The parts the bytes touch are known here too; the page's other title
    // addresses are the same physical page, so the same parts.
    let page = at / PAGE;
    let parts = parts_in(page, at, at.saturating_add(len));
    let before = j.note(page, parts, now);
    for &other in others {
        if other != page {
            j.note(other, parts, now);
        }
    }
    (before, now)
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(j: &mut Journal, pieces: &[(usize, usize)]) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if ask_kernel(j, first, last) {
        j.ahead.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(j, a);
    ask_run(j, b);
}

/// Asks the kernel for the written pages of pages `[first, last)` and notes
/// them in the journal under the ahead asking; false (and nothing changed)
/// if the range is not write-watched memory of one reservation.
fn ask_kernel(j: &mut Journal, first: usize, last: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    if failed {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    let now = j.ahead_asked;
    for k in 0..count {
        let at = j.found[k];
        let (page, part) = (at / PAGE, 1u16 << (at % PAGE / KERNEL_PAGE));
        j.note(page, part, now);
    }
    true
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.ahead
        .iter()
        .any(|&(a, b)| a <= first && last <= b)
        .then(|| (j.ahead_asked, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // Asked ahead with the submission's other ranges: the journal has it.
    if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        return Some((
            j.ahead_asked,
            j.written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
        ));
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // View memory the guards already hold cannot be write-watched: the
    // kernel asking would only fail. Copies of it are flushed many times a
    // frame, each time paying that failed call before the guards answer.
    // Checked and guarded under a single `GUARDS` lock hold, without a
    // lock/unlock round-trip or the failed syscall.
    let guarded = if guarded
        && let Some(written) = try_guard_if_known(first * PAGE, last * PAGE)
    {
        count = 0;
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::WatchKnownView, 1);
            crate::copystat::add(crate::copystat::C::GuardUs, started.elapsed().as_micros() as u64);
        }
        written
    } else {
        let mut granularity = 0;
        // SAFETY: the answer buffer holds `count` entries.
        let failed = unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                j.found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WatchFailed, failed as u64);
        if failed {
            count = 0;
            if !guarded {
                return None;
            }
            #[cfg(feature = "diagnostics")]
            let guarding = std::time::Instant::now();
            let written = guard(first * PAGE, last * PAGE);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
            written?
        } else {
            Vec::new()
        }
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    for k in 0..count {
        let at = j.found[k];
        let (page, part) = (at / PAGE, 1u16 << (at % PAGE / KERNEL_PAGE));
        j.note(page, part, now);
    }
    for page in guarded {
        j.note(page, WHOLE, now);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

impl Tracked {
    /// Takes changed pages of `[from, to)`, cut to the tracked range. A
    /// partial page stays changed so a later take can read its other bytes.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let Some((_, pages)) = writes(from, to, true) else {
            return Vec::new();
        };
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (page, write) in pages {
            if seen.get(&page).is_some_and(|&last| last >= write) {
                continue;
            }
            let (start, end) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
            if start >= end {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from
                && ((page + 1) * PAGE).min(self.end) <= to
            {
                seen.insert(page, write);
            }
            match ranges.last_mut() {
                Some((_, last)) if *last == start => *last = end,
                _ => ranges.push((start, end)),
            }
        }
        ranges
    }
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// Whether the view-asking debug overrides (`VICTORY_VIEW_ASK=1` or
/// `VICTORY_GUARD_ASK=1`) force probing the kernel even for known views.
fn force_view_ask() -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ASK.get_or_init(|| {
        ["VICTORY_VIEW_ASK", "VICTORY_GUARD_ASK"]
            .iter()
            .any(|n| std::env::var_os(n).is_some_and(|v| v == "1"))
    })
}

/// Whether the block holding `start` is view memory the guards already
/// hold (read-only by our hand, or noted absent): asking the kernel's write
/// watch about it can only fail. `VICTORY_VIEW_ASK=1` (or
/// `VICTORY_GUARD_ASK=1`) asks the kernel anyway, as before.
fn known_view(start: usize) -> bool {
    if force_view_ask() {
        return false;
    }
    let first = start / BLOCK;
    let g = guards();
    g.is_known_view_block(first)
}

/// Disjoint coalesced half-open block intervals `[start, end)`, stored as
/// `start -> end` with invariant `start < end` and `prev.end < next.start`.
/// Keeps membership, range-cover, skip-ahead, and range-removal operations
/// `O(log K)` in the number of contiguous runs rather than `O(blocks)`.
struct BlockSet(BTreeMap<usize, usize>);

impl BlockSet {
    const fn new() -> Self {
        Self(BTreeMap::new())
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn contains(&self, block: usize) -> bool {
        self.0.range(..=block).next_back().is_some_and(|(_, &end)| block < end)
    }

    /// True if every block in `[from, to)` is part of one coalesced run.
    fn covers(&self, from: usize, to: usize) -> bool {
        from < to && self.0.range(..=from).next_back().is_some_and(|(_, &end)| end >= to)
    }

    /// If `block` is present, the exclusive end of its contiguous run.
    fn run_end_at(&self, block: usize) -> Option<usize> {
        self.0.range(..=block).next_back().and_then(|(_, &end)| (block < end).then_some(end))
    }

    /// Start of the first interval strictly after `block` (or at `>= block`
    /// if `block` is not inside an interval).
    fn next_start_from(&self, block: usize) -> Option<usize> {
        self.0.range(block..).next().map(|(&start, _)| start)
    }

    /// True if any block in `[from, to)` is present.
    #[cfg_attr(not(test), allow(dead_code))]
    fn intersects(&self, from: usize, to: usize) -> bool {
        if from >= to {
            return false;
        }
        if self.0.range(..=from).next_back().is_some_and(|(_, &end)| end > from) {
            return true;
        }
        self.0.range(from..to).next().is_some()
    }

    /// Total number of blocks in `[from, to)` present in this set.
    fn count_in(&self, from: usize, to: usize) -> usize {
        if from >= to {
            return 0;
        }
        let mut total = 0usize;
        if let Some((_, &end)) = self.0.range(..from).next_back()
            && end > from
        {
            total += end.min(to) - from;
        }
        for (&start, &end) in self.0.range(from..to) {
            total += end.min(to) - start;
        }
        total
    }

    /// Sub-intervals of this set clipped to `[from, to)`.
    fn intersect_ranges(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if from >= to {
            return out;
        }
        if let Some((_, &end)) = self.0.range(..from).next_back()
            && end > from
        {
            out.push((from, end.min(to)));
        }
        for (&start, &end) in self.0.range(from..to) {
            out.push((start, end.min(to)));
        }
        out
    }

    /// Inserts a single `block`.
    fn insert(&mut self, block: usize) {
        if let Some(next) = block.checked_add(1) {
            self.insert_range(block, next);
        }
    }

    fn insert_range(&mut self, mut from: usize, mut to: usize) {
        if from >= to {
            return;
        }
        if let Some((&prev_start, &prev_end)) = self.0.range(..=from).next_back()
            && prev_end >= from
        {
            if prev_end >= to {
                return;
            }
            from = prev_start;
            to = to.max(prev_end);
        }
        while let Some((&s, &e)) = self.0.range(from..=to).next() {
            self.0.remove(&s);
            if e > to {
                to = e;
            }
        }
        self.0.insert(from, to);
    }

    fn insert_sorted_blocks(&mut self, blocks: &[usize]) {
        let mut i = 0;
        while i < blocks.len() {
            let start = blocks[i];
            let mut end = start.saturating_add(1);
            i += 1;
            while i < blocks.len() && blocks[i] <= end {
                if blocks[i] == end {
                    end = end.saturating_add(1);
                }
                i += 1;
            }
            self.insert_range(start, end);
        }
    }

    /// Removes a single `block`; true if it was present.
    fn remove(&mut self, block: usize) -> bool {
        let Some((&s, &e)) = self.0.range(..=block).next_back() else {
            return false;
        };
        if block >= e {
            return false;
        }
        let next = block + 1;
        if s == block {
            self.0.remove(&s);
            if next < e {
                self.0.insert(next, e);
            }
        } else {
            self.0.insert(s, block);
            if next < e {
                self.0.insert(next, e);
            }
        }
        true
    }

    fn remove_range(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        if let Some((&s, &e)) = self.0.range(..from).next_back()
            && e > from
        {
            self.0.insert(s, from);
            if e > to {
                self.0.insert(to, e);
                return;
            }
        }
        while let Some((&s, &e)) = self.0.range(from..to).next() {
            self.0.remove(&s);
            if e > to {
                self.0.insert(to, e);
                break;
            }
        }
    }

    /// Removes all blocks in `[from, to)` and returns the removed sub-intervals
    /// in ascending order.
    fn take_range(&mut self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let mut taken = Vec::new();
        if from >= to {
            return taken;
        }
        if let Some((&s, &e)) = self.0.range(..from).next_back()
            && e > from
        {
            self.0.insert(s, from);
            let cut_end = e.min(to);
            taken.push((from, cut_end));
            if e > to {
                self.0.insert(to, e);
                return taken;
            }
        }
        while let Some((&s, &e)) = self.0.range(from..to).next() {
            self.0.remove(&s);
            let cut_end = e.min(to);
            taken.push((s, cut_end));
            if e > to {
                self.0.insert(to, e);
                break;
            }
        }
        taken
    }
}

struct Guards {
    /// Blocks we made read-only.
    protected: BlockSet,
    /// Of those, the ones not written since a copy took them.
    clean: BlockSet,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: BlockSet,
    /// Subset of `absent` confirmed to lie in `MEM_RESERVE_PLACEHOLDER`
    /// reservations, where pages can only appear through runtime remaps
    /// ([`title_memory_remapped`]): their gaps are skipped without asking.
    placeholder_absent: BlockSet,
    /// Blocks whose physical page was guarded together with its other views
    /// (those views' blocks are in it too): a write to one opens them all.
    /// A write range can be huge (`fread`), so only these are looked up.
    aliased: BlockSet,
    /// With [`view_known`]: blocks made read-only at least once since the
    /// runtime last mapped or unmapped them. They are view memory whether
    /// their guard stands now or a write has opened it.
    viewed: BlockSet,
    /// Blocks a write told to us is under way over ([`KernelWrite`]): no
    /// check guards them again read-only until that write ends. The write
    /// does not fault into the handler (its code is not the title's), so a
    /// guard re-armed under it kills the run where it stands.
    busy: Vec<(usize, usize)>,
}

impl Guards {
    /// Whether `block` is view memory the guards already hold: read-only by
    /// our hand, noted absent, or (with [`view_known`]) seen as a view.
    fn is_known_view_block(&self, block: usize) -> bool {
        self.protected.contains(block)
            || self.absent.contains(block)
            || (view_known() && self.viewed.contains(block))
    }
}

/// Whether a block once guarded stays known as view memory while its
/// guard is open (`VICTORY_VIEW_KNOWN=1`; off by default).
///
/// A range whose first block the title wrote since the last check has
/// that block neither read-only nor noted absent, so nothing says it is a
/// view: the kernel's write watch is asked and fails, and then Windows is
/// asked what the memory is (`is_view`), a walk of the page tables to the
/// end of the pages that share the first one's state. In runs 222 and 225
/// that was 338-450 failed askings a frame and 6.5-7.8 ms a frame of the
/// submitting thread in `VirtualQuery`. With the switch the block is
/// remembered from the time it was first made read-only until the runtime
/// maps or unmaps it ([`title_memory_remapped`]).
fn view_known() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_KNOWN").is_some_and(|v| v == "1"))
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BlockSet::new(),
    clean: BlockSet::new(),
    absent: BlockSet::new(),
    placeholder_absent: BlockSet::new(),
    aliased: BlockSet::new(),
    viewed: BlockSet::new(),
    busy: Vec::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Guards {
    /// Whether a write told to us is under way over `block` ([`KernelWrite`]).
    fn busy(&self, block: usize) -> bool {
        self.busy.iter().any(|&(first, last)| (first..last).contains(&block))
    }
}

/// `VICTORY_ABSENT_ONCE=0`: ask about every absent block at each asking,
/// confirmed placeholder gaps included (as before this change). Unset, or
/// any other value: gaps confirmed to lie in a placeholder reservation are
/// skipped until the runtime tells of a remap ([`title_memory_remapped`]).
fn skip_placeholders() -> bool {
    static SKIP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SKIP.get_or_init(|| std::env::var_os("VICTORY_ABSENT_ONCE").is_none_or(|v| v != "0"))
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    let ask_every = force_view_ask();
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // The known-view question and the guard walk happen under one `GUARDS`
    // lock hold: a frame asks about some 10 GB of view memory and the
    // lock/unlock round-trip per range was part of the "guards" cost.
    let mut g = if !ask_every {
        let mut g = guards();
        let mut known = g.is_known_view_block(first);
        if known && !INSTALLED.load(std::sync::atomic::Ordering::Acquire) {
            drop(g);
            install();
            g = guards();
            known = g.is_known_view_block(first);
        }
        if known {
            g
        } else {
            drop(g);
            if !is_view(start, end) {
                return None;
            }
            install();
            guards()
        }
    } else {
        if !is_view(start, end) {
            return None;
        }
        install();
        guards()
    };
    Some(guard_locked(&mut g, first, last, skip_placeholders()))
}

/// Single-lock fast path for [`writes`]: if `start` is already known view
/// memory, guards `[start, end)` under the same `GUARDS` lock hold without
/// releasing and re-acquiring the mutex. Returns `None` if not yet known (or
/// if the view-ask overrides are on).
fn try_guard_if_known(start: usize, end: usize) -> Option<Vec<usize>> {
    if force_view_ask() {
        return None;
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let mut g = guards();
    if !g.is_known_view_block(first) {
        return None;
    }
    if !INSTALLED.load(std::sync::atomic::Ordering::Acquire) {
        drop(g);
        install();
        g = guards();
        if !g.is_known_view_block(first) {
            return None;
        }
    }
    let written = guard_locked(&mut g, first, last, skip_placeholders());
    drop(g);
    Some(written)
}

/// Exclusive end of the `KernelWrite` range covering `block`, if any.
fn busy_end(busy: &[(usize, usize)], block: usize) -> Option<usize> {
    busy.iter().find(|&&(first, last)| (first..last).contains(&block)).map(|&(_, last)| last)
}

/// Core locked guard walk for blocks `[first, last)`:
/// - Skips contiguous clean and confirmed-absent intervals in `O(log K)` per
///   interval without allocating or touching per-block entries.
/// - Marks already-protected (e.g. alias-guarded) blocks clean in `O(dirty)`
///   without redundant `VirtualProtect` syscalls.
/// - Coalesces contiguous unprotected runs into batched `VirtualProtect` calls
///   under the single lock hold.
fn guard_locked(g: &mut Guards, first: usize, last: usize, skip_placeholders: bool) -> Vec<usize> {
    if first >= last || g.clean.covers(first, last) {
        // Nothing written, mapped or new since the range was last taken.
        return Vec::new();
    }
    let mut written = Vec::new();
    let mut run: Option<(usize, usize)> = None;
    let mut uncommitted_until = first;
    let mut cursor = first;
    while cursor < last {
        // 0. A write told to us is under way over it (`KernelWrite`): not
        //    taken again, and not reported until it ends, when its bytes are
        //    all there. A run of blocks up to it is guarded as it was.
        if !g.busy.is_empty()
            && let Some(until) = busy_end(&g.busy, cursor)
        {
            if let Some((rf, rt)) = run.take() {
                protect(g, rf, rt);
            }
            cursor = until.min(last);
            continue;
        }

        // 1. Skip an entire contiguous clean interval in one step.
        if let Some(clean_end) = g.clean.run_end_at(cursor) {
            if let Some((rf, rt)) = run.take() {
                protect(g, rf, rt);
            }
            cursor = clean_end.min(last);
            continue;
        }

        // 2. Skip an entire contiguous absent interval where valid.
        if let Some(absent_end) = g.absent.run_end_at(cursor) {
            let seg_end = absent_end.min(last);
            // Found uncommitted and confirmed to be a placeholder: pages
            // appear there only through a runtime remap, which is told
            // ([`title_memory_remapped`]), so Windows is not asked at all.
            if skip_placeholders
                && let Some(placeholder_end) = g.placeholder_absent.run_end_at(cursor)
            {
                let skip_to = placeholder_end.min(seg_end);
                if let Some((rf, rt)) = run.take() {
                    protect(g, rf, rt);
                }
                cursor = skip_to;
                continue;
            }
            // Ordinary commits need no runtime mapping notification. Reuse
            // this asking's query for the following blocks in the same region.
            if cursor >= uncommitted_until {
                uncommitted_until = uncommitted_end(cursor);
            }
            if cursor < uncommitted_until {
                let skip_to = uncommitted_until.min(seg_end);
                if let Some((rf, rt)) = run.take() {
                    protect(g, rf, rt);
                }
                cursor = skip_to;
                continue;
            }
            // Pages appeared at `cursor` without a runtime remap notification.
            g.absent.remove(cursor);
            g.placeholder_absent.remove(cursor);
        }

        // 3. Dirty/unguarded span from `cursor` to the next clean, absent or
        //    busy boundary.
        let mut dirty_end = last;
        if let Some(start) = g.clean.next_start_from(cursor) {
            dirty_end = dirty_end.min(start);
        }
        if let Some(start) = g.absent.next_start_from(cursor) {
            dirty_end = dirty_end.min(start);
        }
        for &(busy_from, _) in &g.busy {
            if busy_from > cursor {
                dirty_end = dirty_end.min(busy_from);
            }
        }
        const { assert!(BLOCK == PAGE) };
        written.extend(cursor..dirty_end);

        let mut d = cursor;
        while d < dirty_end {
            if let Some(prot_end) = g.protected.run_end_at(d) {
                // Already read-only (e.g. guarded through an alias view): mark
                // clean without re-calling `VirtualProtect`.
                let clean_to = prot_end.min(dirty_end);
                if let Some((rf, rt)) = run.take() {
                    protect(g, rf, rt);
                }
                g.clean.insert_range(d, clean_to);
                d = clean_to;
            } else {
                let unprot_to = g.protected.next_start_from(d).map_or(dirty_end, |s| s.min(dirty_end));
                match &mut run {
                    Some((_, r_to)) if *r_to == d => *r_to = unprot_to,
                    Some((r_from, r_to)) => {
                        let (rf, rt) = (*r_from, *r_to);
                        run = Some((d, unprot_to));
                        protect(g, rf, rt);
                    }
                    None => run = Some((d, unprot_to)),
                }
                d = unprot_to;
            }
        }
        cursor = dirty_end;
    }

    if let Some((rf, rt)) = run.take() {
        protect(g, rf, rt);
    }
    written
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing mapped
/// counts as a view only as far as its holes are reserved placeholders
/// (`MEM_RESERVE`), which is what the runtime maps again; free memory
/// (`MEM_FREE`) is no view of the title's, and counting it as one tracked
/// it (blocks noted absent, its pages reported written) — freed and then
/// allocated again, it brought those notes to the process heap and its
/// pages into the write journal, where a later allocation inherited them
/// (`free_memory_is_not_tracked`).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    let mut placeholder = false;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole: a placeholder the runtime maps again, or free memory,
        // which is no view. On to what follows it.
        placeholder |= info.State == MEM_RESERVE;
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return placeholder;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// splitting along the OS's own memory regions if they span views or
/// uncommitted holes, leaving out any that fail, and noting those not
/// committed (unmapped) absent. The same physical pages at every other view
/// of them are made read-only too: a write through one of those is this
/// memory written, and only a fault can say so.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
fn protect(g: &mut Guards, from: usize, to: usize) {
    protect_here(g, from, to);
    if !view_guard() {
        return;
    }
    let (sources, all_aliased, peers) =
        crate::inplace::aliases_in_range(from.saturating_mul(BLOCK), to.saturating_mul(BLOCK));
    if sources == 0 {
        return;
    }
    #[cfg(feature = "diagnostics")]
    ALIASES[0].fetch_add(sources, std::sync::atomic::Ordering::Relaxed);
    g.aliased.insert_sorted_blocks(&all_aliased);
    // Coalesce unprotected peer alias blocks into contiguous runs.
    let mut i = 0;
    while i < peers.len() {
        let block = peers[i];
        if g.protected.contains(block) || (!g.busy.is_empty() && g.busy(block)) {
            i += 1;
            continue;
        }
        let run_start = block;
        let mut run_end = block + 1;
        i += 1;
        while i < peers.len()
            && peers[i] == run_end
            && !g.protected.contains(peers[i])
            && (g.busy.is_empty() || !g.busy(peers[i]))
        {
            run_end += 1;
            i += 1;
        }
        protect_alias_run(g, run_start, run_end);
    }
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Records `[from, to)` as guarded in `g` (and `clean` when `mark_clean`),
/// clearing any absent state there, so that `protected` and `absent` stay
/// disjoint by construction: a block that becomes read-only can no longer be
/// skipped as a hole, and `open` sees every guard it must give back.
fn note_protected(g: &mut Guards, from: usize, to: usize, mark_clean: bool) {
    if !g.absent.is_empty() {
        g.absent.remove_range(from, to);
    }
    if !g.placeholder_absent.is_empty() {
        g.placeholder_absent.remove_range(from, to);
    }
    g.protected.insert_range(from, to);
    if mark_clean {
        g.clean.insert_range(from, to);
    }
    if view_known() {
        g.viewed.insert_range(from, to);
    }
}

/// Makes contiguous alias blocks `[from, to)` read-only in a single coalesced
/// `VirtualProtect` (splitting along OS VM regions if they span views): another
/// view of pages a copy has just taken through its own view, so that a write
/// through this one faults too. It is not noted clean. The copies of title
/// memory at this view have not taken the pages: noted clean, the block would
/// not be reported written at their next check and they would keep the bytes
/// from before the write (`guard` reports a block that is read-only and not
/// clean as written, then notes it clean). A block that is clean stays so: had
/// the pages been written since its copies took them, `open` would have opened
/// it.
fn protect_alias_run(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY};
    if os_virtual_protect(from, to, PAGE_READONLY) {
        note_protected(g, from, to, false);
        return;
    }
    if to <= from + 1 {
        return;
    }
    let mut at = from;
    while at < to {
        let Some(addr) = at.checked_mul(BLOCK) else {
            break;
        };
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = os_virtual_query(addr, &mut info);
        if n == 0 {
            at += 1;
            continue;
        }
        let region_end = (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK;
        if info.State != MEM_COMMIT {
            at = region_end.min(to).max(at + 1);
            continue;
        }
        let sub_to = region_end.min(to).max(at + 1);
        if os_virtual_protect(at, sub_to, PAGE_READONLY) {
            note_protected(g, at, sub_to, false);
        } else if sub_to > at + 1 {
            for block in at..sub_to {
                if os_virtual_protect(block, block + 1, PAGE_READONLY) {
                    note_protected(g, block, block + 1, false);
                }
            }
        }
        at = sub_to;
    }
}

/// [`protect`] for one range, without the other views of its pages.
fn protect_here(g: &mut Guards, from: usize, to: usize) {
    if g.busy.is_empty() {
        protect_free(g, from, to);
        return;
    }
    // A block a write told to us is under way over (`KernelWrite`) is left
    // as it is: guarding it again read-only would make that write fault in
    // code that is not the title's, and the run would die there. It is
    // guarded again by the first check after the write ends.
    let mut cuts: Vec<(usize, usize)> = g
        .busy
        .iter()
        .map(|&(first, last)| (first.max(from), last.min(to)))
        .filter(|&(first, last)| first < last)
        .collect();
    cuts.sort_unstable();
    let mut at = from;
    for (first, last) in cuts {
        if first > at {
            protect_free(g, at, first);
        }
        at = at.max(last);
    }
    if at < to {
        protect_free(g, at, to);
    }
}

/// [`protect_here`] of a range no write is under way over: one
/// `VirtualProtect` per committed OS region, no failed calls over
/// uncommitted holes.
fn protect_free(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_FREE, MEM_IMAGE, MEM_PRIVATE, MEM_RESERVE, MEMORY_BASIC_INFORMATION,
        PAGE_READONLY,
    };
    // Only view memory is made read-only. A range can run past the views
    // into other committed memory: with the structure builder on, the
    // 4 MiB 0x400000-0x800000 (a window that begins in mapped memory, so
    // `is_view` says yes) had every committed private block in it made
    // read-only block by block, the process heap's among them. The heap's
    // own code then faulted while holding its lock, the faults waited for
    // the guard lock, and the world froze. Committed private memory and
    // images are never guarded (as Marvisak/VIctory's 280b150 decides region
    // by region).
    let mut at = from;
    while at < to {
        let Some(addr) = at.checked_mul(BLOCK) else {
            break;
        };
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = os_virtual_query(addr, &mut info);
        if n == 0 {
            at += 1;
            continue;
        }
        let base_end = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if info.State == MEM_COMMIT && matches!(info.Type, MEM_PRIVATE | MEM_IMAGE) {
            at = base_end.div_ceil(BLOCK).clamp(at + 1, to);
            continue;
        }
        if matches!(info.State, MEM_RESERVE | MEM_FREE) {
            // Nothing committed: a placeholder the runtime maps later, or
            // free memory. Noted absent, with no failed protection call for
            // each block of it.
            let end = (base_end / BLOCK).min(to);
            if end > at {
                note_absent_span(g, at, end);
                at = end;
            } else {
                // Partly committed block (starts reserved, a 4 KiB page
                // committed inside it): cannot be guarded and is no hole.
                let _ = handle_protect_failure(g, at, to);
                at += 1;
            }
            continue;
        }
        // Committed view memory: one protection call for the whole region.
        let region_end = base_end.div_ceil(BLOCK).clamp(at + 1, to);
        if os_virtual_protect(at, region_end, PAGE_READONLY) {
            note_protected(g, at, region_end, true);
        } else if region_end > at + 1 {
            let mut block = at;
            while block < region_end {
                if os_virtual_protect(block, block + 1, PAGE_READONLY) {
                    note_protected(g, block, block + 1, true);
                    block += 1;
                } else {
                    block = handle_protect_failure(g, block, region_end).max(block + 1);
                }
            }
        } else {
            at = handle_protect_failure(g, at, to).max(at + 1);
            continue;
        }
        at = region_end;
    }
}

#[cfg(test)]
thread_local! {
    static OS_VIRTUAL_PROTECT_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static OS_VIRTUAL_QUERY_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Changes page protection for contiguous blocks `[from, to)` in one syscall.
fn os_virtual_protect(from: usize, to: usize, protect_flags: u32) -> bool {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    if from >= to {
        return true;
    }
    let Some(addr) = from.checked_mul(BLOCK) else {
        return false;
    };
    let Some(len) = (to - from).checked_mul(BLOCK) else {
        return false;
    };
    #[cfg(test)]
    OS_VIRTUAL_PROTECT_CALLS.with(|c| c.set(c.get() + 1));
    let mut old = 0;
    // SAFETY: title view memory; only its page protection changes.
    unsafe { VirtualProtect(addr as *const _, len, protect_flags, &mut old) != 0 }
}

/// Queries OS virtual memory metadata for `addr`.
fn os_virtual_query(
    addr: usize,
    info: &mut windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION,
) -> usize {
    use windows_sys::Win32::System::Memory::VirtualQuery;
    #[cfg(test)]
    OS_VIRTUAL_QUERY_CALLS.with(|c| c.set(c.get() + 1));
    // SAFETY: a valid out pointer of the right size.
    unsafe { VirtualQuery(addr as *const _, info, size_of_val(info)) }
}

/// Records an uncommitted span `[from, to)` as absent, and as
/// placeholder-absent for every sub-region `NtQueryVirtualMemory` confirms
/// to be a `MEM_RESERVE_PLACEHOLDER` reservation (advancing by region size).
fn note_absent_span(g: &mut Guards, from: usize, to: usize) {
    if from >= to {
        return;
    }
    g.absent.insert_range(from, to);
    let mut b = from;
    while b < to {
        let (is_placeholder, region_end) = placeholder_region_info(b);
        let next = region_end.min(to).max(b + 1);
        if is_placeholder {
            g.placeholder_absent.insert_range(b, next);
        }
        b = next;
    }
}

/// Handles a single block that failed `VirtualProtect(..., PAGE_READONLY)`.
/// Returns the exclusive end block handled (at least `block + 1`) so callers
/// advance past any whole uncommitted span recorded in `g.absent`.
fn handle_protect_failure(g: &mut Guards, block: usize, limit: usize) -> usize {
    #[cfg(feature = "diagnostics")]
    let error = std::io::Error::last_os_error();
    let end = uncommitted_end(block).min(limit);
    if block < end {
        note_absent_span(g, block, end);
        end
    } else {
        // Diagnostics: a committed block that will not be made read-only is
        // reported written at every asking and never known to the guards
        // (one in 1,024 said).
        #[cfg(feature = "diagnostics")]
        {
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed).is_multiple_of(1024) {
                tracing::info!(
                    "a block of title memory could not be made read-only (one in 1024): {:#x}: {error}",
                    block.saturating_mul(BLOCK)
                );
            }
        }
        block + 1
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION};
    let Some(addr) = block.checked_mul(BLOCK) else {
        return block;
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let n = os_virtual_query(addr, &mut info);
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// `MEMORY_REGION_INFORMATION` (ntmmapi): the kinds of a region, as
/// `NtQueryVirtualMemory` reports them. Only the fields read here.
#[repr(C)]
#[derive(Default)]
struct Region {
    allocation_base: *mut std::ffi::c_void,
    allocation_protect: u32,
    region_type: u32,
    region_size: usize,
    commit_size: usize,
}

#[link(name = "ntdll")]
unsafe extern "system" {
    fn NtQueryVirtualMemory(
        process: windows_sys::Win32::Foundation::HANDLE,
        base: *const std::ffi::c_void,
        class: u32,
        info: *mut std::ffi::c_void,
        len: usize,
        returned: *mut usize,
    ) -> i32;
}

/// Queries `NtQueryVirtualMemory(MemoryRegionInformationEx)` at `block`.
/// Returns `(is_placeholder, region_end_block)` where `region_end_block` is
/// at least `block + 1` so callers can advance past an entire OS region
/// (placeholder or ordinary reservation) in one syscall. A placeholder
/// reservation is memory in which pages come to be only by mapping a view, or
/// making an allocation, in its place, which the runtime does and tells of
/// ([`title_memory_remapped`]). An ordinary reservation can be committed by
/// anybody, and free memory taken by anybody, with no word to us:
/// `VirtualQuery` does not tell the three apart, this question does.
fn placeholder_region_info(block: usize) -> (bool, usize) {
    /// The `MemoryRegionInformationEx` information class.
    const MEMORY_REGION_INFORMATION_EX: u32 = 7;
    /// `MEMORY_REGION_INFORMATION_TYPE.PlaceholderReservation`, bit 8 of the
    /// region type word.
    const PLACEHOLDER_RESERVATION: u32 = 1 << 8;
    let Some(addr) = block.checked_mul(BLOCK) else {
        return (false, block.saturating_add(1));
    };
    let mut region = Region::default();
    // SAFETY: the answer's room and its size; the kernel checks the address.
    let status = unsafe {
        NtQueryVirtualMemory(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            addr as *const _,
            MEMORY_REGION_INFORMATION_EX,
            (&raw mut region).cast(),
            size_of::<Region>(),
            std::ptr::null_mut(),
        )
    };
    if status >= 0 {
        let end = (region.allocation_base as usize).saturating_add(region.region_size) / BLOCK;
        let is_placeholder = region.region_type & PLACEHOLDER_RESERVATION != 0 && end > block;
        (is_placeholder, end.max(block.saturating_add(1)))
    } else {
        (false, block.saturating_add(1))
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded and given write access (a
/// block that could not be stays in the guards). The same physical pages at
/// every other view of them are opened too: the write that faulted (or is
/// ours, `crate::host::copy_mapped`) goes through one view, and every
/// guard over the pages must copy them again.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    if start >= end {
        return false;
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // The common fault (the handler, one block, no alias of its pages): no
    // allocation, one syscall, and a failure leaves the block in the guards.
    if last == first + 1 && !g.aliased.contains(first) && g.protected.contains(first) {
        if os_virtual_protect(first, last, PAGE_READWRITE) {
            g.protected.remove(first);
            g.clean.remove(first);
            return true;
        }
        tracing::error!(
            "a guarded block of title memory could not be given write access: {:#x} (block {first:#x}): {}",
            first * BLOCK,
            std::io::Error::last_os_error()
        );
        return false;
    }
    let aliased_runs = if g.aliased.is_empty() { Vec::new() } else { g.aliased.take_range(first, last) };
    let peer_blocks = if aliased_runs.is_empty() {
        Vec::new()
    } else {
        crate::inplace::aliases_for_blocks(&aliased_runs)
    };
    let primary_runs = g.protected.intersect_ranges(first, last);
    let opened = open_runs(g, &primary_runs);
    // The same physical pages at every other view of them, in coalesced runs.
    let mut alias_runs: Vec<(usize, usize)> = Vec::new();
    for &block in &peer_blocks {
        g.aliased.remove(block);
        if g.protected.contains(block) {
            match alias_runs.last_mut() {
                Some((_, run_end)) if *run_end == block => *run_end = block + 1,
                _ => alias_runs.push((block, block + 1)),
            }
        }
    }
    let alias_opened = open_runs(g, &alias_runs);
    #[cfg(not(feature = "diagnostics"))]
    let _ = alias_opened;
    #[cfg(feature = "diagnostics")]
    ALIASES[1].fetch_add(alias_opened as u64, std::sync::atomic::Ordering::Relaxed);
    opened > 0
}

/// [`open`] for one range, without the other views of its pages. True if a
/// block of the range was guarded and given write access: a block still
/// read-only is not counted, and stays in the guards for the next fault.
fn open_here(g: &mut Guards, start: usize, end: usize) -> bool {
    if start >= end {
        return false;
    }
    let runs = g.protected.intersect_ranges(start / BLOCK, end.div_ceil(BLOCK));
    open_runs(g, &runs) > 0
}

/// Gives write access back to `runs` of guarded blocks, coalesced (one
/// `VirtualProtect` per contiguous OS region); returns how many blocks were
/// given write access. A block whose protection cannot be restored stays in
/// `protected` and in `clean`, so the next write there faults into the
/// handler and is looked at again.
fn open_runs(g: &mut Guards, runs: &[(usize, usize)]) -> usize {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE};
    let mut opened = 0;
    for &(from, to) in runs {
        if from >= to {
            continue;
        }
        if os_virtual_protect(from, to, PAGE_READWRITE) {
            g.protected.remove_range(from, to);
            g.clean.remove_range(from, to);
            opened += to - from;
            continue;
        }
        if to <= from + 1 {
            tracing::error!(
                "a guarded block of title memory could not be given write access: {:#x} (block {from:#x}): {}",
                from * BLOCK,
                std::io::Error::last_os_error()
            );
            continue;
        }
        let mut at = from;
        while at < to {
            let Some(addr) = at.checked_mul(BLOCK) else {
                break;
            };
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            let n = os_virtual_query(addr, &mut info);
            if n == 0 {
                if os_virtual_protect(at, at + 1, PAGE_READWRITE) {
                    g.protected.remove(at);
                    g.clean.remove(at);
                    opened += 1;
                }
                at += 1;
                continue;
            }
            let region_end = ((info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK)
                .min(to)
                .max(at + 1);
            if info.State != MEM_COMMIT {
                at = region_end;
                continue;
            }
            if os_virtual_protect(at, region_end, PAGE_READWRITE) {
                g.protected.remove_range(at, region_end);
                g.clean.remove_range(at, region_end);
                opened += region_end - at;
            } else if region_end > at + 1 {
                for block in at..region_end {
                    if os_virtual_protect(block, block + 1, PAGE_READWRITE) {
                        g.protected.remove(block);
                        g.clean.remove(block);
                        opened += 1;
                    } else {
                        tracing::error!(
                            "a guarded block of title memory could not be given write access: {:#x} (block {block:#x}): {}",
                            block * BLOCK,
                            std::io::Error::last_os_error()
                        );
                    }
                }
            } else {
                tracing::error!(
                    "a guarded block of title memory could not be given write access: {:#x} (block {at:#x}): {}",
                    at * BLOCK,
                    std::io::Error::last_os_error()
                );
            }
            at = region_end;
        }
    }
    opened
}

/// Restores `PAGE_READWRITE` for contiguous blocks `[from, to)` in a single
/// coalesced `VirtualProtect`, splitting along OS VM regions if the run spans
/// multiple views and skipping uncommitted gaps without per-block failures.
#[cfg_attr(not(test), allow(dead_code))]
fn unprotect_run_os(from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE};
    if os_virtual_protect(from, to, PAGE_READWRITE) || to <= from + 1 {
        return;
    }
    let mut at = from;
    while at < to {
        let Some(addr) = at.checked_mul(BLOCK) else {
            break;
        };
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = os_virtual_query(addr, &mut info);
        if n == 0 {
            at += 1;
            continue;
        }
        let region_end = (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK;
        let sub_to = region_end.min(to).max(at + 1);
        if info.State != MEM_COMMIT {
            at = sub_to;
            continue;
        }
        if !os_virtual_protect(at, sub_to, PAGE_READWRITE) && sub_to > at + 1 {
            for block in at..sub_to {
                os_virtual_protect(block, block + 1, PAGE_READWRITE);
            }
        }
        at = sub_to;
    }
}

/// Before a write into title memory `[at, at + len)` the guards do not see
/// (the kernel's, which does not fault into our handler): the guarded blocks
/// there are open while this lives, and no check guards them again under the
/// write. Dropped, the blocks it opened are made read-only again (where
/// they are still view memory) and stay not clean: the next check reports
/// the write to every copy of them, which takes the bytes that are there.
#[must_use = "the write must happen while this lives"]
pub struct KernelWrite {
    first: usize,
    last: usize,
    /// Blocks it opened and must guard again.
    opened: Vec<usize>,
}

/// Opens the guarded blocks of `[at, at + len)` and keeps them open until
/// the result is dropped ([`KernelWrite`]).
fn hold(at: usize, len: usize) -> KernelWrite {
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    let mut g = guards();
    let mut opened = Vec::new();
    if first < last {
        // Before the first byte lands: from here no check guards these
        // blocks again, whatever it finds written in them.
        g.busy.push((first, last));
        let before: Vec<usize> = g
            .protected
            .intersect_ranges(first, last)
            .into_iter()
            .flat_map(|(from, to)| from..to)
            .collect();
        open(&mut g, at, at.saturating_add(len));
        opened = before.into_iter().filter(|&block| !g.protected.contains(block)).collect();
    }
    KernelWrite { first, last, opened }
}

impl Drop for KernelWrite {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualProtect, VirtualQuery,
        };
        let mut g = guards();
        if let Some(i) = g.busy.iter().position(|&r| r == (self.first, self.last)) {
            g.busy.swap_remove(i);
        }
        for &block in &self.opened {
            // View memory only, as `protect_here` decides block by block: a
            // block the runtime remapped to private memory meanwhile must
            // not be made read-only (the heap's own writes then fault).
            // SAFETY: plain data filled in by the query, of the right size.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
            if n == 0 || info.State != MEM_COMMIT || info.Type != MEM_MAPPED {
                g.clean.remove(block);
                continue;
            }
            let mut old = 0;
            // SAFETY: title view memory; only its protection changes.
            if unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0 {
                // Read-only again, still not clean: the next check reports
                // the write. Guarded and clean, the copies would keep the
                // bytes from before it.
                g.protected.insert(block);
            }
            g.clean.remove(block);
        }
    }
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory. The blocks are left open (noted written):
/// this is the telling of a write, not a guard over one.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// Before a write into title memory `[at, at + len)` that goes around the
/// guards and does not fault into our handler — DirectStorage writing
/// physical pages through a view of its own, at the addresses the title has
/// them mapped (`victory_hle::xmem`'s `page_window`, `write_physical`): the
/// guarded blocks there must be open **before** the first byte lands, since
/// the write is not the title's code and a fault in it is not ours to take.
/// The result holds them open to the end of the write. The texture pass is
/// not told here: what the write wrote is said when it is done
/// ([`title_memory_written`], which stays for the count).
pub fn will_write(at: usize, len: usize) -> KernelWrite {
    hold(at, len)
}

/// [`will_write`], and the placed textures there are noted as about to
/// change (the kernel's `ReadFile` and `fread` into the title's own memory).
pub fn writing_held(at: usize, len: usize) -> KernelWrite {
    let guard = hold(at, len);
    crate::descriptor::host::remapped(at, len);
    guard
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished.
/// Unset or `guards`: bit 1 alone. `0`: nothing, counted only (the default
/// until run 199). `1`: bits 1 and 2. `slots`: bit 2 alone. `all`:
/// everything.
/// (Bit 1 by default: in every run 10,000 to 50,000 guarded blocks of host
/// copies lay under such writes and kept the bytes from before; runs 197
/// and 198 opened them, 9,766 and 10,796 blocks, with nothing worse for
/// it. Bit 2 makes texture slots due and is the one that flooded the
/// queue in an earlier form: still asked for by name.)
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("1") => 3,
        Ok("slots") => 2,
        Ok("all") => 7,
        _ => 1,
    })
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished.
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Such a write
/// was told to the watch before it began ([`will_write`]), which held its
/// blocks open; this counts it and does what [`page_writes`] says once its
/// bytes are there. Otherwise a read to a plain address has finished, of
/// which [`writing`] was told before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        if mode & 4 == 0 {
            return;
        }
    }
    {
        let mut g = guards();
        let guarded = g.protected.count_in(at / BLOCK, end.div_ceil(BLOCK)) as u64;
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 && guarded != 0 {
            open(&mut g, at, end);
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again — and are, the pages noted written, which also has the next check
/// guard the new view (`protect`).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    let mut g = guards();
    for &(at, n) in &ranges {
        if open_here(&mut g, at, at.saturating_add(n)) {
            #[cfg(feature = "diagnostics")]
            ALIASES[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>, std::time::Instant);

impl Drop for Remapping {
    fn drop(&mut self) {
        // From asking for the lock until it is let go (the frame log).
        crate::framelog::add(crate::framelog::T::RemapUs, self.1.elapsed().as_micros() as u64);
    }
}

/// The runtime is about to map, unmap or move physical pages of title
/// memory: until what it returns is dropped, no read of title memory in
/// place is under way and none begins ([`title_memory_reading`]). Taken
/// around the whole change, the unmapping and the mapping again.
pub fn title_memory_remapping() -> Remapping {
    let asked = std::time::Instant::now();
    let held = MAPPING.write().unwrap_or_else(|e| e.into_inner());
    crate::framelog::add(crate::framelog::T::Remaps, 1);
    crate::framelog::add(crate::framelog::T::RemapWaitUs, asked.elapsed().as_micros() as u64);
    Remapping(held, asked)
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>);

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    Reading(MAPPING.read().unwrap_or_else(|e| e.into_inner()))
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping (or waits to), for a caller that has another
/// way to read and would rather not wait.
pub fn title_memory_reading_now() -> Option<Reading> {
    match MAPPING.try_read() {
        Ok(guard) => Some(Reading(guard)),
        Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(e.into_inner())),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE};
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    if len == 0 {
        return;
    }
    let mut g = guards();
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    // Nothing stays noted about the range: no hole, nothing aliased.
    g.absent.remove_range(first, last);
    g.placeholder_absent.remove_range(first, last);
    g.aliased.remove_range(first, last);
    for (from, to) in g.protected.intersect_ranges(first, last) {
        for block in from..to {
            // Forgetting a block that is still read-only by our hand would
            // leave it read-only with no set knowing it: a page the guards no
            // longer have, which the handler therefore never opens. That state
            // is what killed the run (rt2, `long25`: a 256 KiB write of ours —
            // `CopyMemoryToMemoryX` — over `PAGE_READONLY`, `MEM_MAPPED`, in no
            // set). It arises when the runtime's notification arrives while the
            // view is still mapped and guarded (`xmem`'s `unmap` told us before
            // `UnmapViewOfFile2`, fixed there) or when a remap elsewhere in the
            // range leaves the block as it was. Give the write access back
            // before dropping the block from the guards, so "read-only and
            // unregistered" cannot exist.
            // SAFETY: plain data filled in by the query, of the right size.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            let n = os_virtual_query(block * BLOCK, &mut info);
            if n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READONLY {
                // SAFETY: the block was made read-only by us; only its
                // protection changes, and only back to write access.
                if !os_virtual_protect(block, block + 1, PAGE_READWRITE) {
                    // Left in the guards (still read-only, still known) so
                    // the next write there faults into the handler and is
                    // looked at again, instead of dying unannounced. Noted
                    // not clean: what the range shows is new, and the copies
                    // must take it.
                    g.clean.remove(block);
                    tracing::error!(
                        "a guarded block of title memory could not be given write access on remap: {:#x} (block {block:#x}): {}",
                        block * BLOCK,
                        std::io::Error::last_os_error()
                    );
                    continue;
                }
            }
            g.protected.remove(block);
            g.clean.remove(block);
        }
    }
    // What the range shows is new: no longer known to be a view.
    g.viewed.remove_range(first, last);
}

/// Whether the handler that takes writes to guarded blocks is registered
/// ([`install`]): the guard walks ask before making blocks read-only.
static INSTALLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
pub(crate) fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        // A fault of one of our own copies (`crate::safecopy`), and where its
        // failing exit is: asked first, so that a page that is not there
        // fails the copy instead of reaching the crash reporter.
        let ours = crate::safecopy::failing_exit(r.ExceptionAddress as usize);
        let fail = move |exit: usize| {
            // SAFETY: the context the system passes, to go on from.
            let context = unsafe { (*info).ContextRecord };
            // SAFETY: as above.
            unsafe { (*context).Rip = exit as u64 };
            EXCEPTION_CONTINUE_EXECUTION
        };
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return match ours {
                Some(exit) => fail(exit),
                None => EXCEPTION_CONTINUE_SEARCH,
            };
        }
        let at = r.ExceptionInformation[1];
        // A WRITE of ours is opened and retried here, never sent to the
        // failing exit: the copy is to put our bytes, and stopping it at the
        // guard would leave the title's old bytes in place, silently (the
        // decodings `shadow` warns of). Only a block the guards will not open
        // goes to the exit.
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            crate::framelog::add(crate::framelog::T::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            // The run dies here. One line, only on this path: what the
            // guards make of the fault and what Windows says of the page,
            // so that the next one is a proof rather than a signature.
            use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
            let block = at / BLOCK;
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
            tracing::error!(
                "a write to {at:#x} faults and the guards let it die: block {block:#x} protected {} clean {} absent {} aliased {} viewed {}; \
                 VirtualQuery {n}: state {:#x} protect {:#x} type {:#x} base {:#x} region {:#x}; one of our copies: {ours:?}",
                g.protected.contains(block),
                g.clean.contains(block),
                g.absent.contains(block),
                g.aliased.contains(block),
                g.viewed.contains(block),
                info.State,
                info.Protect,
                info.Type,
                info.BaseAddress as usize,
                info.RegionSize
            );
            match ours {
                Some(exit) => fail(exit),
                None => EXCEPTION_CONTINUE_SEARCH,
            }
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| {
        unsafe {
            AddVectoredExceptionHandler(1, Some(on_write));
        }
        INSTALLED.store(true, std::sync::atomic::Ordering::Release);
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// Forgets what the journal knows of `[start, end)`: it runs for the
    /// whole process, and the pages another test wrote to may be the ones
    /// this allocation lands on.
    fn forget(start: usize, end: usize) {
        let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        for page in start / PAGE..end.div_ceil(PAGE) {
            j.written.remove(&page);
            j.parts.remove(&page);
        }
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        // The journal is the process's, not this allocation's: start from
        // nothing known of these pages, whatever an earlier test wrote.
        forget(start, end);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A page the kernel's answer names by its 4 KiB offset keeps that part
    /// through the asking that stands for the submission (as `writes`
    /// answers it): a refresh cuts the copy there (`VICTORY_TAKE_PARTS=1`);
    /// off, the page, as before.
    #[test]
    fn a_page_asked_ahead_keeps_the_parts_it_was_written_in() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                2 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        // The journal is the process's: nothing known of these pages before
        // the first asking, so the page reported below is this test's write.
        forget(p, p + 2 * PAGE);
        ask_once_end();
        let (taken, _) = writes(p, p + 2 * PAGE, false).expect("watched");
        let page = p / PAGE + 1;
        // SAFETY: committed pages of ours: the fourth 4 KiB part of page 1.
        unsafe { ((page * PAGE + 3 * KERNEL_PAGE + 7) as *mut u8).write(1) };
        // One asking for the submission's two ranges.
        ask_ahead(&[(p, p + PAGE), (p + PAGE, p + 2 * PAGE)]);
        let (_, pages) = writes(p, p + 2 * PAGE, false).expect("watched");
        assert!(pages.iter().any(|&(p, _)| p == page));
        let parts = parts_written(&[(page, taken)]);
        if parts_on() {
            assert_eq!(parts, [1 << 3]);
            // A write of ours into part of a page is those parts, not WHOLE.
            let at = page * PAGE + 2 * KERNEL_PAGE + 1;
            assert_eq!(parts_in(page, at, at + 2 * KERNEL_PAGE - 1), 0b1100);
        } else {
            assert_eq!(parts, [WHOLE]);
        }
        ask_ahead_done();
        // The journal outlives the allocation: the next test may get these
        // pages again, and would find them written.
        forget(p, p + 2 * PAGE);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// Free memory is not tracked: tracked, it would be guarded once
    /// something is allocated there (the process heap: its writes then
    /// fault into the handler while a thread holds the guard lock).
    #[test]
    fn free_memory_is_not_tracked() {
        use windows_sys::Win32::System::Memory::{MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: a reservation of ours, released at once: free memory.
        let r = unsafe { VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(r, 0);
        // SAFETY: our reservation.
        unsafe { VirtualFree(r as *mut _, 0, MEM_RELEASE) };
        let tracked = track(r, 2 * PAGE, true).is_some();
        // Other tests allocate meanwhile: only memory still free counts.
        // SAFETY: plain data filled in by the query, of the right size.
        let still_free = unsafe {
            let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            VirtualQuery(r as *const _, &mut info, size_of_val(&info)) != 0
                && info.State == MEM_FREE
                && info.BaseAddress as usize + info.RegionSize >= r + 2 * PAGE
        };
        assert!(!(tracked && still_free), "free memory");
    }

    /// A range that begins in a view and runs into committed private
    /// memory guards the view only: the private block (the process heap's,
    /// say) stays writable.
    #[test]
    fn committed_private_memory_beside_a_view_is_not_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MapViewOfFile3, PAGE_NOACCESS,
            UnmapViewOfFile2, VirtualAlloc2, VirtualQuery,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of two blocks, split; a pagefile
        // section's block mapped over the first, private memory committed
        // over the second.
        let (p, section, view) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                2 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            assert_ne!(VirtualFree(p as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER), 0);
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            let view = MapViewOfFile3(
                section,
                GetCurrentProcess(),
                p as *const _,
                0,
                BLOCK,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(view.Value as usize, p);
            let private = VirtualAlloc2(
                GetCurrentProcess(),
                (p + BLOCK) as *const _,
                BLOCK,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_eq!(private, p + BLOCK);
            (p, section, view)
        };
        title_memory_remapped(p, BLOCK);
        assert!(is_view(p, p + 2 * BLOCK), "it begins in a view");
        assert!(guard(p, p + 2 * BLOCK).is_some());
        let protection = |at: usize| {
            // SAFETY: plain data filled in by the query, of the right size.
            unsafe {
                let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                assert_ne!(VirtualQuery(at as *const _, &mut info, size_of_val(&info)), 0);
                info.Protect
            }
        };
        assert_eq!(protection(p), windows_sys::Win32::System::Memory::PAGE_READONLY, "the view is guarded");
        assert_eq!(protection(p + BLOCK), PAGE_READWRITE, "the private block is not");
        // SAFETY: committed private memory of ours, writable.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        title_memory_remapped(p, 2 * BLOCK);
        // SAFETY: our view, section and private block.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), view, 0);
            VirtualFree((p + BLOCK) as *mut _, 0, MEM_RELEASE);
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        let writing = writing(view + 100, 13);
        assert!(read(&mut file));
        drop(writing);
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages
    /// (`writing`): the other views' next checks report the pages written.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        drop(writing(other + BLOCK, BLOCK));
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        drop(writing(other + BLOCK / 2, BLOCK));
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write told to the watch *before* it lands (`will_write`, the path
    /// of DirectStorage's page writes): its blocks are open while the guard
    /// lives — a check under way neither guards them again (the write's own
    /// code would die on a read-only block) nor reports them half written —
    /// and when the guard falls they are read-only again, not clean, and
    /// the write is reported once.
    #[test]
    fn a_write_told_before_it_lands_is_held_open_until_its_guard_falls() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        let writing = will_write(guarded + BLOCK, BLOCK);
        assert!(writable(guarded + BLOCK), "open before the write");
        assert!(!writable(guarded), "the block beside it stays guarded");
        // A check while the write is under way leaves its block alone.
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "under way: nothing taken");
        assert!(writable(guarded + BLOCK), "still open");
        // SAFETY: the block we opened.
        unsafe { ((guarded + BLOCK + 8) as *mut u8).write(0x5a) };
        drop(writing);
        assert!(!writable(guarded + BLOCK), "guarded again when the guard falls");
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]), "the write is reported once");
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// The runtime says a block was remapped while it is still a view we
    /// made read-only: `xmem`'s `unmap` used to tell the guards **before**
    /// `UnmapViewOfFile2` (run rt2, `long25`), and a remap elsewhere in the
    /// range leaves the block read-only all the same. Forgetting the block
    /// must open it; otherwise it stays read-only with no set knowing it
    /// (`VirtualQuery`: `PAGE_READONLY`, `MEM_MAPPED`, no guards) and the
    /// next write of ours over it — 256 KiB in `long25`, from
    /// `CopyMemoryToMemoryX` — dies there, out of the handler's reach.
    #[test]
    fn a_remapped_guarded_block_is_left_writable() {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        let (section, guarded, other) = two_views(BLOCK);
        assert_eq!(guard(guarded, guarded + BLOCK), Some(vec![guarded / PAGE]));
        // The block is read-only by our hand and known to the guards.
        assert!(!writable(guarded), "guarded read-only");
        assert!(guards().protected.contains(guarded / BLOCK));
        // The notification alone, the view still mapped: the interleaving
        // `unmap` had (the notification before the unmapping).
        title_memory_remapped(guarded, BLOCK);
        // SAFETY: plain data filled in by the query, of the right size.
        let protection = unsafe {
            let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            assert_ne!(VirtualQuery(guarded as *const _, &mut info, size_of_val(&info)), 0);
            info.Protect
        };
        assert_eq!(
            protection,
            windows_sys::Win32::System::Memory::PAGE_READWRITE,
            "a block the runtime remapped is not left read-only and unregistered"
        );
        assert!(writable(guarded));
        assert!(!guards().protected.contains(guarded / BLOCK));
        drop_views(section, &[guarded, other], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        drop(writing(top, PAGE));
        title_memory_remapped(top, usize::MAX);
    }

    /// Serializes tests that assert exact process-wide syscall counter deltas.
    fn syscall_turn() -> std::sync::MutexGuard<'static, ()> {
        static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
        TURN.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn syscall_counts() -> (u64, u64) {
        (
            OS_VIRTUAL_PROTECT_CALLS.with(|c| c.get()),
            OS_VIRTUAL_QUERY_CALLS.with(|c| c.get()),
        )
    }

    #[test]
    fn block_set_interval_invariants() {
        let mut s = BlockSet::new();
        assert!(s.is_empty());
        s.insert_range(10, 20);
        s.insert_range(30, 40);
        assert!(s.covers(10, 20) && !s.covers(10, 21) && !s.covers(20, 30));
        assert_eq!(s.run_end_at(15), Some(20));
        assert_eq!(s.run_end_at(20), None);
        assert_eq!(s.next_start_from(20), Some(30));
        assert_eq!(s.count_in(15, 35), 10);
        assert_eq!(s.intersect_ranges(15, 35), vec![(15, 20), (30, 35)]);
        // Bridge adjacent intervals into a single coalesced run.
        s.insert_range(20, 30);
        assert!(s.covers(10, 40));
        assert_eq!(s.0.len(), 1);
        // Punch a hole in the middle.
        let taken = s.take_range(15, 25);
        assert_eq!(taken, vec![(15, 25)]);
        assert_eq!(s.0.len(), 2);
        assert!(s.covers(10, 15) && s.covers(25, 40) && !s.intersects(15, 25));
        // Single-block insert & remove.
        s.insert(20);
        assert!(s.contains(20) && s.remove(20) && !s.remove(20));
        // Coalesce a sorted block list, duplicates and all.
        s.insert_sorted_blocks(&[15, 16, 17, 100, 101, 102]);
        assert!(s.covers(10, 18) && s.covers(100, 103));
    }

    /// Helper managing a placeholder reservation with independently mapped
    /// section views and split 64 KiB placeholder holes.
    struct SparsePlaceholder {
        base: usize,
        blocks: usize,
        section: *mut std::ffi::c_void,
        mapped: Vec<(usize, usize, windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS)>,
        split_blocks: Vec<bool>,
    }

    impl SparsePlaceholder {
        fn new(blocks: usize) -> Self {
            use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
            use windows_sys::Win32::System::Memory::{
                CreateFileMappingW, MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, VirtualAlloc2,
            };
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            // SAFETY: fresh placeholder reservation + pagefile section, freed in Drop.
            let (base, section) = unsafe {
                let base = VirtualAlloc2(
                    GetCurrentProcess(),
                    std::ptr::null(),
                    blocks * BLOCK,
                    MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                    PAGE_NOACCESS,
                    std::ptr::null_mut(),
                    0,
                ) as usize;
                assert_ne!(base, 0);
                let section = CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE,
                    0,
                    (blocks * BLOCK) as u32,
                    std::ptr::null(),
                );
                assert!(!section.is_null());
                (base, section)
            };
            Self {
                base,
                blocks,
                section,
                mapped: Vec::new(),
                split_blocks: vec![false; blocks],
            }
        }

        /// Splits `[from, to)` into individual 64 KiB placeholder regions
        /// (the layout xmem leaves where every hole is its own placeholder).
        fn split_holes_per_block(&mut self, from: usize, to: usize) {
            use windows_sys::Win32::System::Memory::MEM_PRESERVE_PLACEHOLDER;
            for b in from..to {
                // SAFETY: splits a placeholder inside our reservation.
                unsafe {
                    VirtualFree(
                        (self.base + b * BLOCK) as *mut _,
                        BLOCK,
                        MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                    );
                }
                self.split_blocks[b] = true;
            }
        }

        /// Maps contiguous blocks `[from, to)` as a single view of the section.
        fn map_run(&mut self, from: usize, to: usize) {
            use windows_sys::Win32::System::Memory::{
                MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER, MapViewOfFile3,
            };
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            let len = (to - from) * BLOCK;
            let addr = self.base + from * BLOCK;
            // SAFETY: splits the placeholder for `[from, to)` and replaces it
            // with a view of the section.
            let view = unsafe {
                VirtualFree(addr as *mut _, len, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                MapViewOfFile3(
                    self.section,
                    GetCurrentProcess(),
                    addr as *const _,
                    (from * BLOCK) as u64,
                    len,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, addr);
            self.mapped.push((from, to, view));
        }
    }

    impl Drop for SparsePlaceholder {
        fn drop(&mut self) {
            use windows_sys::Win32::Foundation::CloseHandle;
            use windows_sys::Win32::System::Memory::{MEM_PRESERVE_PLACEHOLDER, UnmapViewOfFile2};
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            title_memory_remapped(self.base, self.blocks * BLOCK);
            // SAFETY: unmaps every view and releases all placeholder slices we own.
            unsafe {
                for &(_, _, view) in &self.mapped {
                    UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
                }
                for b in 0..self.blocks {
                    VirtualFree((self.base + b * BLOCK) as *mut _, 0, MEM_RELEASE);
                }
                CloseHandle(self.section);
            }
        }
    }

    /// Sparse placeholder holes (including individually split 64 KiB
    /// placeholders) are skipped in `O(intervals)` with zero `VirtualProtect`
    /// syscalls on subsequent passes until remapped, and the holes cost no
    /// `VirtualQuery` either.
    #[test]
    fn sparse_placeholder_gaps_are_skipped_without_probing_or_protecting() {
        let _turn = syscall_turn();
        const TOTAL: usize = 64;
        let mut sp = SparsePlaceholder::new(TOTAL);
        sp.map_run(0, 8);
        sp.split_holes_per_block(8, 24);
        sp.map_run(24, 32);
        sp.split_holes_per_block(32, 56);
        sp.map_run(56, 64);
        let (start, end) = (sp.base, sp.base + TOTAL * BLOCK);
        let base_page = start / PAGE;

        // First guard pass reports all 64 blocks and protects only the 3
        // mapped views (no per-block `VirtualProtect` over the 40 holes).
        let (vp0, _) = syscall_counts();
        let first = guard(start, end).expect("view");
        let (vp1, _) = syscall_counts();
        assert_eq!(first.len(), TOTAL);
        assert!(
            vp1 - vp0 <= 4,
            "expected <=4 VirtualProtect calls (3 mapped views), got {}",
            vp1 - vp0
        );

        // Steady state: 0 VirtualProtect AND 0 VirtualQuery across all 40
        // split holes.
        let (vp2, vq2) = syscall_counts();
        for _ in 0..16 {
            assert_eq!(guard(start, end), Some(Vec::new()));
        }
        let (vp3, vq3) = syscall_counts();
        assert_eq!(vp3 - vp2, 0, "steady-state guard must issue 0 VirtualProtect calls");
        assert_eq!(vq3 - vq2, 0, "steady-state guard must issue 0 VirtualQuery calls on placeholders");

        // Write 1 byte inside mapped block 25: only block 25 is reported and
        // re-protected.
        // SAFETY: block 25 is mapped in `sp`.
        unsafe { ((start + 25 * BLOCK + 16) as *mut u8).write_volatile(0x5A) };
        let (vp4, vq4) = syscall_counts();
        assert_eq!(guard(start, end), Some(vec![base_page + 25]));
        let (vp5, vq5) = syscall_counts();
        assert_eq!(vp5 - vp4, 1, "only the 1 dirty block should be re-protected");
        // One query at most: that of the block being re-armed, asked because
        // committed private memory beside a view is never guarded. None for
        // the 39 placeholder holes.
        assert!(vq5 - vq4 <= 1, "placeholder holes must stay skipped without VirtualQuery");

        // Map one previously absent placeholder block (block 12) and tell the
        // runtime's remap.
        sp.map_run(12, 13);
        title_memory_remapped(start + 12 * BLOCK, BLOCK);
        let (vp6, vq6) = syscall_counts();
        assert_eq!(guard(start, end), Some(vec![base_page + 12]));
        let (vp7, vq7) = syscall_counts();
        assert_eq!(vp7 - vp6, 1, "only the newly mapped block should be protected");
        // One query at most: that of the block being re-armed, asked because
        // committed private memory beside a view is never guarded. None for
        // the remaining 39 placeholder holes.
        assert!(vq7 - vq6 <= 1, "remaining placeholder holes must stay skipped");
        assert_eq!(guard(start, end), Some(Vec::new()));
    }

    /// Contiguous runs coalesce both guard (`PAGE_READONLY`) and unguard
    /// (`PAGE_READWRITE`) into single syscalls per contiguous OS view,
    /// including partial unguards and multi-VAD boundaries.
    #[test]
    fn contiguous_runs_coalesce_guard_and_unguard_syscalls() {
        let _turn = syscall_turn();
        const N: usize = 32;
        let mut sp = SparsePlaceholder::new(N);
        sp.map_run(0, N);
        let (start, end) = (sp.base, sp.base + N * BLOCK);
        let base_page = start / PAGE;

        // 1. Guard all 32 contiguous blocks -> 1 VirtualProtect call.
        let (vp0, _) = syscall_counts();
        let all = guard(start, end).expect("view");
        let (vp1, _) = syscall_counts();
        assert_eq!(all.len(), N);
        assert_eq!(vp1 - vp0, 1, "32 contiguous blocks must coalesce into 1 VirtualProtect");

        // 2. Partial unguard of the middle 16 blocks [8..24) -> 1 call.
        let (vp2, _) = syscall_counts();
        writing(start + 8 * BLOCK, 16 * BLOCK);
        let (vp3, _) = syscall_counts();
        assert_eq!(vp3 - vp2, 1, "opening 16 contiguous blocks must coalesce into 1 VirtualProtect");

        // 3. Re-guarding [0..32) skips clean [0..8) and [24..32) and protects
        //    [8..24) in 1 call (plus the one query that keeps committed
        //    private memory beside a view from being guarded).
        let (vp4, vq4) = syscall_counts();
        let dirty = guard(start, end).expect("view");
        let (vp5, vq5) = syscall_counts();
        let expected: Vec<usize> = (base_page + 8..base_page + 24).collect();
        assert_eq!(dirty, expected);
        assert_eq!(vp5 - vp4, 1, "O(dirty) re-arm must re-protect [8..24) in 1 VirtualProtect");
        assert_eq!(vq5 - vq4, 1);

        // 4. Multi-VAD boundary: two adjacent 16-block views inside one
        //    32-block range take one call each (our walk classifies the OS
        //    region before it touches it, so there is no failed bulk attempt).
        let mut sp2 = SparsePlaceholder::new(N);
        sp2.map_run(0, 16);
        sp2.map_run(16, 32);
        let (s2, e2) = (sp2.base, sp2.base + N * BLOCK);

        let (vp6, _) = syscall_counts();
        assert_eq!(guard(s2, e2).expect("view").len(), N);
        let (vp7, _) = syscall_counts();
        assert_eq!(
            vp7 - vp6,
            2,
            "2 adjacent 16-block VADs must take 2 per-VAD VirtualProtect calls, not 32"
        );

        let (vp8, _) = syscall_counts();
        writing(s2, N * BLOCK);
        let (vp9, _) = syscall_counts();
        assert_eq!(
            vp9 - vp8,
            3,
            "opening 2 adjacent 16-block VADs must coalesce per VAD (1 failed bulk + 2 calls, not 32)"
        );
    }

    /// Dual-aliased views coalesce alias protection and unprotection into
    /// single syscalls per contiguous run, and the second view's first guard
    /// marks already-protected blocks clean with 0 VirtualProtect calls.
    #[test]
    fn aliased_views_coalesce_protect_open_and_o_dirty_rearm() {
        let _turn = syscall_turn();
        const N: usize = 32;
        const SIZE: usize = N * BLOCK;
        let (section, one, two) = two_views(SIZE);

        // Guarding `one` protects `one` (1 call) + coalesced peer `two` (1
        // call) = 2 calls (not 33).
        let (vp0, _) = syscall_counts();
        assert_eq!(guard(one, one + SIZE).expect("view").len(), N);
        let (vp1, _) = syscall_counts();
        assert_eq!(vp1 - vp0, 2, "primary + alias view should take 2 coalesced VirtualProtect calls");

        // Guarding `two` reports all 32 blocks as newly seen for `two` and
        // marks them clean with ZERO VirtualProtect calls: already read-only.
        let (vp2, _) = syscall_counts();
        assert_eq!(guard(two, two + SIZE).expect("view").len(), N);
        let (vp3, _) = syscall_counts();
        assert_eq!(vp3 - vp2, 0, "already-protected alias blocks must require 0 VirtualProtect calls");

        // Opening all 32 blocks of `one` opens both views in 2 coalesced
        // calls (not 64).
        let (vp4, _) = syscall_counts();
        writing(one, SIZE);
        let (vp5, _) = syscall_counts();
        assert_eq!(vp5 - vp4, 2, "opening aliased 32-block range must coalesce into 2 VirtualProtect calls");

        // Re-guard both views cleanly, then fault a single block through `two`.
        assert_eq!(guard(one, one + SIZE).expect("view").len(), N);
        assert_eq!(guard(two, two + SIZE).expect("view").len(), N);
        // SAFETY: block 10 of `two` is mapped; the handler opens it on both views.
        unsafe { ((two + 10 * BLOCK + 4) as *mut u8).write_volatile(0xAB) };
        let (vp6, _) = syscall_counts();
        assert_eq!(guard(one, one + SIZE), Some(vec![one / PAGE + 10]));
        let (vp7, _) = syscall_counts();
        assert_eq!(vp7 - vp6, 2, "re-guarding 1 dirty aliased block takes 2 VirtualProtect calls");
        assert_eq!(guard(two, two + SIZE), Some(vec![two / PAGE + 10]));
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));

        drop_views(section, &[one, two], SIZE);
    }

    /// Concurrent writers (faulting stores and `writing()` calls) racing
    /// concurrent `writes()` readers across aliased views stay crash-free and
    /// report every final write once the writers finish.
    #[test]
    fn concurrent_writers_and_guard_readers_stay_coherent() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        const N: usize = 16;
        const SIZE: usize = N * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let _ = writes(one, one + SIZE, true).expect("view");
        let _ = writes(two, two + SIZE, true).expect("view");

        let done = Arc::new(AtomicBool::new(false));
        let mut threads = Vec::new();
        for tid in 0..4usize {
            let view = if tid % 2 == 0 { one } else { two };
            threads.push(std::thread::spawn(move || {
                for step in 0..200usize {
                    let b = (tid * 4 + step) % N;
                    let addr = view + b * BLOCK + (tid * 16);
                    if step % 7 == 0 {
                        writing(addr, 64);
                    }
                    // SAFETY: committed mapped view; the handler opens guarded
                    // blocks before the write completes.
                    unsafe { (addr as *mut u32).write_volatile((tid as u32) << 16 | step as u32) };
                }
            }));
        }
        for rid in 0..2usize {
            let done = done.clone();
            let view = if rid == 0 { one } else { two };
            threads.push(std::thread::spawn(move || {
                let mut loops = 0usize;
                while !done.load(Ordering::Acquire) && loops < 400 {
                    let res = writes(view, view + SIZE, true);
                    assert!(res.is_some());
                    loops += 1;
                }
            }));
        }
        for t in threads.drain(..4) {
            t.join().unwrap();
        }
        done.store(true, Ordering::Release);
        for t in threads {
            t.join().unwrap();
        }

        // After the writers settle, both views drain clean and a fresh write
        // on each block is caught.
        let _ = guard(one, one + SIZE).expect("view");
        let _ = guard(two, two + SIZE).expect("view");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        for b in 0..N {
            // SAFETY: mapped view `two`.
            unsafe { ((two + b * BLOCK) as *mut u8).write_volatile(0xEE) };
        }
        let reported_one = guard(one, one + SIZE).expect("view");
        let reported_two = guard(two, two + SIZE).expect("view");
        assert_eq!(reported_one.len(), N);
        assert_eq!(reported_two.len(), N);

        drop_views(section, &[one, two], SIZE);
    }

    /// Deterministic side-by-side benchmark comparing the pre-change baseline
    /// (`BTreeSet` per-block walk + per-block `VirtualProtect` + per-block
    /// `VirtualQuery`) against this file's `guard`/`open` on live Windows
    /// placeholders and aliased views.
    #[test]
    fn benchmark_guard_coalesce_and_absent_skip() {
        use std::collections::BTreeSet;
        use std::time::Instant;
        use windows_sys::Win32::System::Memory::{
            MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualProtect,
            VirtualQuery,
        };

        let _turn = syscall_turn();

        // A local baseline replicating the pre-change BTreeSet algorithm
        // (per-block walk, per-block fallback VirtualProtect, per-block
        // VirtualQuery on split placeholders, per-block open).
        struct LegacyGuards {
            protected: BTreeSet<usize>,
            clean: BTreeSet<usize>,
            absent: BTreeSet<usize>,
            vp_calls: u64,
            vq_calls: u64,
        }
        impl LegacyGuards {
            fn new() -> Self {
                Self {
                    protected: BTreeSet::new(),
                    clean: BTreeSet::new(),
                    absent: BTreeSet::new(),
                    vp_calls: 0,
                    vq_calls: 0,
                }
            }
            fn uncommitted_end(&mut self, block: usize) -> usize {
                self.vq_calls += 1;
                let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
                if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
                    (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
                } else {
                    block
                }
            }
            fn protect_here(&mut self, from: usize, to: usize) {
                let mut old = 0;
                self.vp_calls += 1;
                if unsafe { VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, PAGE_READONLY, &mut old) } != 0 {
                    self.protected.extend(from..to);
                    self.clean.extend(from..to);
                    return;
                }
                for block in from..to {
                    self.vp_calls += 1;
                    if unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0 {
                        self.protected.insert(block);
                        self.clean.insert(block);
                    } else if block < self.uncommitted_end(block) {
                        self.absent.insert(block);
                    }
                }
            }
            fn guard_walk(&mut self, start: usize, end: usize) -> usize {
                let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
                let clean: Vec<usize> = self.clean.range(first..last).copied().collect();
                if clean.len() == last - first {
                    return 0;
                }
                let mut written = 0usize;
                let mut next_clean = 0;
                let mut uncommitted_until = first;
                let mut run: Option<usize> = None;
                for block in first..=last {
                    while next_clean < clean.len() && clean[next_clean] < block {
                        next_clean += 1;
                    }
                    let is_clean = next_clean < clean.len() && clean[next_clean] == block;
                    let mut open = block < last && !is_clean;
                    if open && self.absent.contains(&block) {
                        if block >= uncommitted_until {
                            uncommitted_until = self.uncommitted_end(block);
                        }
                        open = block >= uncommitted_until;
                        if open {
                            self.absent.remove(&block);
                        }
                    }
                    if open {
                        written += 1;
                        if self.protected.contains(&block) {
                            self.clean.insert(block);
                        } else {
                            run.get_or_insert(block);
                            continue;
                        }
                    }
                    if let Some(from) = run.take() {
                        self.protect_here(from, block);
                    }
                }
                written
            }
            fn open_range(&mut self, start: usize, end: usize) {
                let blocks: Vec<usize> = self.protected.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
                for block in blocks {
                    self.protected.remove(&block);
                    self.clean.remove(&block);
                    let mut old = 0;
                    self.vp_calls += 1;
                    unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
                }
            }
        }

        // Workload 1: sparse placeholder reservation (192 blocks = 12 MiB:
        // 128 split 64 KiB placeholder holes + 64 mapped blocks in 4 views),
        // 60 submissions with 4 dirty blocks each.
        const SPARSE_BLOCKS: usize = 192;
        const SUBMISSIONS: usize = 60;
        let mut sp = SparsePlaceholder::new(SPARSE_BLOCKS);
        sp.map_run(0, 16);
        sp.split_holes_per_block(16, 48);
        sp.map_run(48, 64);
        sp.split_holes_per_block(64, 96);
        sp.map_run(96, 112);
        sp.split_holes_per_block(112, 144);
        sp.map_run(144, 160);
        sp.split_holes_per_block(160, 192);
        let (s_start, s_end) = (sp.base, sp.base + SPARSE_BLOCKS * BLOCK);

        // Baseline run on the sparse placeholder workload.
        let mut legacy = LegacyGuards::new();
        let t0 = Instant::now();
        let _ = legacy.guard_walk(s_start, s_end);
        for sub in 0..SUBMISSIONS {
            let dirty_base = match sub % 4 {
                0 => 0,
                1 => 48,
                2 => 96,
                _ => 144,
            };
            legacy.open_range(s_start + dirty_base * BLOCK, s_start + (dirty_base + 4) * BLOCK);
            let w = legacy.guard_walk(s_start, s_end);
            assert_eq!(w, 4);
        }
        let legacy_sparse_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let legacy_sparse_vp = legacy.vp_calls;
        let legacy_sparse_vq = legacy.vq_calls;
        legacy.open_range(s_start, s_end);

        // This file's run on the exact same sparse placeholder memory.
        let (vp0, vq0) = syscall_counts();
        let t1 = Instant::now();
        let _ = guard(s_start, s_end).expect("view");
        for sub in 0..SUBMISSIONS {
            let dirty_base = match sub % 4 {
                0 => 0,
                1 => 48,
                2 => 96,
                _ => 144,
            };
            writing(s_start + dirty_base * BLOCK, 4 * BLOCK);
            let w = guard(s_start, s_end).expect("view");
            assert_eq!(w.len(), 4);
        }
        let opt_sparse_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let (vp1, vq1) = syscall_counts();
        let opt_sparse_vp = vp1 - vp0;
        let opt_sparse_vq = vq1 - vq0;

        // Workload 2: contiguous dual-aliased 64-block view (4 MiB x 2), 30
        // open + guard cycles.
        const ALIAS_BLOCKS: usize = 64;
        const ALIAS_SIZE: usize = ALIAS_BLOCKS * BLOCK;
        const CYCLES: usize = 30;
        let (section, one, two) = two_views(ALIAS_SIZE);

        let mut legacy_alias = LegacyGuards::new();
        let t2 = Instant::now();
        let _ = legacy_alias.guard_walk(one, one + ALIAS_SIZE);
        for b in 0..ALIAS_BLOCKS {
            let mut old = 0;
            legacy_alias.vp_calls += 1;
            unsafe { VirtualProtect((two + b * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) };
            legacy_alias.protected.insert(two / BLOCK + b);
        }
        for _ in 0..CYCLES {
            legacy_alias.open_range(one, one + ALIAS_SIZE);
            legacy_alias.open_range(two, two + ALIAS_SIZE);
            let _ = legacy_alias.guard_walk(one, one + ALIAS_SIZE);
            for b in 0..ALIAS_BLOCKS {
                let mut old = 0;
                legacy_alias.vp_calls += 1;
                unsafe { VirtualProtect((two + b * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) };
                legacy_alias.protected.insert(two / BLOCK + b);
            }
        }
        let legacy_alias_ms = t2.elapsed().as_secs_f64() * 1000.0;
        let legacy_alias_vp = legacy_alias.vp_calls;
        legacy_alias.open_range(one, one + ALIAS_SIZE);
        legacy_alias.open_range(two, two + ALIAS_SIZE);

        let (vp2, _) = syscall_counts();
        let t3 = Instant::now();
        assert_eq!(guard(one, one + ALIAS_SIZE).expect("view").len(), ALIAS_BLOCKS);
        for _ in 0..CYCLES {
            writing(one, ALIAS_SIZE);
            assert_eq!(guard(one, one + ALIAS_SIZE).expect("view").len(), ALIAS_BLOCKS);
        }
        let opt_alias_ms = t3.elapsed().as_secs_f64() * 1000.0;
        let (vp3, _) = syscall_counts();
        let opt_alias_vp = vp3 - vp2;

        drop_views(section, &[one, two], ALIAS_SIZE);

        eprintln!(
            "[bench watch::guard] sparse(192blk, 128 holes, {SUBMISSIONS} subs): \
             VirtualProtect {legacy_sparse_vp} -> {opt_sparse_vp} ({:.1}x), \
             VirtualQuery {legacy_sparse_vq} -> {opt_sparse_vq} ({:.1}x), \
             wall {legacy_sparse_ms:.3} ms -> {opt_sparse_ms:.3} ms",
            legacy_sparse_vp as f64 / opt_sparse_vp.max(1) as f64,
            legacy_sparse_vq as f64 / opt_sparse_vq.max(1) as f64,
        );
        eprintln!(
            "[bench watch::guard] aliased(64blk x2, {CYCLES} cycles): \
             VirtualProtect {legacy_alias_vp} -> {opt_alias_vp} ({:.1}x), \
             wall {legacy_alias_ms:.3} ms -> {opt_alias_ms:.3} ms",
            legacy_alias_vp as f64 / opt_alias_vp.max(1) as f64,
        );

        // Deterministic syscall-reduction assertions (never timing-flaky).
        assert!(
            opt_sparse_vp * 3 <= legacy_sparse_vp,
            "sparse VirtualProtect calls should drop >=3x: legacy={legacy_sparse_vp}, opt={opt_sparse_vp}"
        );
        assert!(
            opt_sparse_vq * 30 <= legacy_sparse_vq,
            "sparse VirtualQuery calls should drop >=30x: legacy={legacy_sparse_vq}, opt={opt_sparse_vq}"
        );
        assert!(
            opt_alias_vp * 30 <= legacy_alias_vp,
            "aliased VirtualProtect calls should drop >=30x: legacy={legacy_alias_vp}, opt={opt_alias_vp}"
        );
    }

    /// Ordinary non-placeholder `SEC_RESERVE` holes are recorded in `absent`
    /// only (never `placeholder_absent`), so a silent
    /// `VirtualAlloc(..., MEM_COMMIT)` without a runtime notification is
    /// detected immediately by the default `guard()`.
    #[test]
    fn ordinary_reserve_is_not_placeholder_and_silent_commit_is_detected() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: mapped by this test.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: reserved pagefile section, mapped once.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let v = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!v.Value.is_null());
            View(v)
        };
        let start = view.0.Value as usize;
        let base_block = start / BLOCK;
        // Commit block 0 only; blocks 1 and 2 stay ordinary MEM_RESERVE.
        // SAFETY: inside our reserved view.
        assert_eq!(unsafe { VirtualAlloc(start as *const _, BLOCK, MEM_COMMIT, PAGE_READWRITE) } as usize, start);

        assert_eq!(guard(start, start + SIZE), Some(vec![base_block, base_block + 1, base_block + 2]));
        {
            let g = guards();
            assert!(g.absent.covers(base_block + 1, base_block + 3));
            assert!(
                !g.placeholder_absent.intersects(base_block + 1, base_block + 3),
                "ordinary SEC_RESERVE must never enter placeholder_absent"
            );
        }
        assert_eq!(guard(start, start + SIZE), Some(Vec::new()));

        // Silent VirtualAlloc(MEM_COMMIT) on block 2, no runtime notification.
        // SAFETY: inside our reserved view.
        assert_eq!(
            unsafe { VirtualAlloc((start + 2 * BLOCK) as *const _, BLOCK, MEM_COMMIT, PAGE_READWRITE) } as usize,
            start + 2 * BLOCK
        );
        assert_eq!(guard(start, start + SIZE), Some(vec![base_block + 2]));
        assert_eq!(guard(start, start + SIZE), Some(Vec::new()));
    }

    /// `protected` and `absent` stay disjoint whenever a previously absent
    /// span becomes protected, so a later write there is never silently
    /// dropped.
    #[test]
    fn protected_and_absent_sets_remain_disjoint_after_protect() {
        let mut sp = SparsePlaceholder::new(4);
        sp.map_run(0, 1);
        sp.split_holes_per_block(1, 4);
        let (start, end) = (sp.base, sp.base + 4 * BLOCK);
        let base_block = start / BLOCK;

        // First guard records blocks 1..4 in absent & placeholder_absent.
        assert_eq!(guard(start, end).expect("view").len(), 4);
        {
            let g = guards();
            assert!(g.absent.covers(base_block + 1, base_block + 4));
            assert!(g.placeholder_absent.covers(base_block + 1, base_block + 4));
        }

        // Map block 2 without telling the runtime, then note it protected as
        // `protect_free` would if a commit raced with the absent marking.
        sp.map_run(2, 3);
        {
            let mut g = guards();
            assert!(os_virtual_protect(
                base_block + 2,
                base_block + 3,
                windows_sys::Win32::System::Memory::PAGE_READONLY
            ));
            note_protected(&mut g, base_block + 2, base_block + 3, true);
            assert!(g.protected.contains(base_block + 2));
            assert!(!g.absent.contains(base_block + 2), "protected block must be evicted from absent");
            assert!(
                !g.placeholder_absent.contains(base_block + 2),
                "protected block must be evicted from placeholder_absent"
            );
        }

        // A write fault to block 2 must open it and be reported by the default
        // `guard()` (it would have been silently dropped had block 2 stayed in
        // placeholder_absent).
        // SAFETY: block 2 is mapped in `sp`.
        unsafe { ((start + 2 * BLOCK + 8) as *mut u8).write_volatile(0x33) };
        assert_eq!(guard(start, end), Some(vec![base_block + 2]));
        assert_eq!(guard(start, end), Some(Vec::new()));
    }

    /// `unprotect_run_os` across a partially unmapped span skips the
    /// uncommitted gap via `VirtualQuery` (`MEM_COMMIT` check) without falling
    /// back to per-block failing `VirtualProtect` syscalls, and restores
    /// `PAGE_READWRITE` on both surrounding committed views.
    #[test]
    fn unprotect_run_os_skips_uncommitted_gap_without_per_block_failures() {
        let _turn = syscall_turn();
        const N: usize = 20;
        let mut sp = SparsePlaceholder::new(N);
        sp.map_run(0, 8);
        sp.split_holes_per_block(8, 12);
        sp.map_run(12, 20);
        let (start, end) = (sp.base, sp.base + N * BLOCK);
        let base_block = start / BLOCK;

        assert_eq!(guard(start, end).expect("view").len(), N);
        assert!(!writable(start) && !writable(start + 12 * BLOCK));

        let (vp0, _) = syscall_counts();
        unprotect_run_os(base_block, base_block + N);
        let (vp1, _) = syscall_counts();
        assert!(
            vp1 - vp0 <= 3,
            "expected <=3 VirtualProtect calls (1 failed bulk + 2 committed views), got {}",
            vp1 - vp0
        );
        assert!(writable(start) && writable(start + 7 * BLOCK));
        assert!(writable(start + 12 * BLOCK) && writable(start + 19 * BLOCK));
        // Clean up guard state before Drop.
        title_memory_remapped(start, N * BLOCK);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// A change of mapping waits for a read in place that is under way, and
    /// a read in place waits for a change of mapping.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading();
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping();
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping();
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading();
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }
}
