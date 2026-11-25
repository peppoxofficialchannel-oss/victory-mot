//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one buffer for all lists, entries only ever go from empty to
//!   filled;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the copies exist from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs; the
//!   addresses a dispatch is given as root views are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end); all list upload memory (`crate::list::upload`), where
//! the builder records the tables its stages read through pointers
//! (InitHeaders' table of structure addresses); and the title addresses in
//! the block of upload memory a root view points into (the leaf
//! rehydration's geometry descriptions: its index and vertex buffers).
//! shortcut: what a stage reaches only through pointers into other title
//! memory is not made ready unless something else named it; the fault
//! count in the log says whether that happens. The whole answer is the
//! extent of the title's allocation around an address, which the memory
//! module of the layer knows and this one does not ask yet.
//!
//! shortcut: a copy that is numbered is kept for the rest of the run, and
//! there are at most [`COPIES`] of them (4 MiB windows mostly: 4 GiB at
//! most). Every dispatch of such a shader brings all numbered copies up to
//! date and, if it can store through an address, declares all of them
//! written (the bytes are compared on the CPU afterwards, as for any
//! unordered access view): a dirty-page mask written by the shader would
//! bound both.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer). A
/// listed frame of the opening cutscene names about 50 windows; the
/// structure builder's dispatches in the user-save scenario 1,125 in their
/// first frame and about 1,450 in six minutes, low dwords of address pairs
/// left out (run rt-dry1).
const COPIES: u32 = 2047;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Arc<Shadow>>,
    /// Per host view heap (by address): the first slot of its run, and how
    /// many copies have their view there.
    heaps: HashMap<usize, (u32, u32)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let make = |dwords: u64, what: &str| {
            let (buffer, data) = host.shared_buffer(dwords * 4)?;
            crate::host::name(&buffer, || format!("title memory by address: {what}"));
            // SAFETY: a fresh mapping of at least that many bytes.
            unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data.cast::<u32>(), gpu))
        };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: make(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: make(u64::from(MEMORY_FAULTS), "faults")?,
            copies: Vec::new(),
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
        })
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's (`why`: for the trace, `crate::bvhtrace`): the index
    /// in `copies` of the copy that holds the page, if one does.
    fn map(&mut self, address: u64, why: &str) -> Option<usize> {
        #[cfg(not(feature = "diagnostics"))]
        let _ = why;
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return None;
        }
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: as above.
        let had = unsafe { entry.read() };
        if had != 0 {
            return Some(had as usize - 1);
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return None;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return None;
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let shadow = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))?;
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return None;
        }
        let n = match self.copies.iter().position(|c| Arc::ptr_eq(c, &shadow)) {
            Some(n) => n,
            None if self.copies.len() < COPIES as usize => {
                self.copies.push(shadow.clone());
                let n = self.copies.len() - 1;
                #[cfg(feature = "diagnostics")]
                crate::bvhtrace::line(|| {
                    let (first, len) = shadow.span();
                    format!(
                        "N {} {first:#x}+{len:#x} window={} at {address:#x} why={why}",
                        n + 1,
                        shadow.is_window()
                    )
                });
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB.
                // shortcut: a page of a title buffer inside this one that
                // has a copy of its own is given this copy too.
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    // SAFETY: inside the directory's mapping.
                    let other = unsafe { self.directory.1.add((at / PAGE) as usize * 2) };
                    // SAFETY: as above; the offset first, as below.
                    unsafe {
                        if other.read() == 0 {
                            other.add(1).write_volatile((at - first) as u32);
                            other.write_volatile(n as u32 + 1);
                        }
                    }
                    at += PAGE;
                }
                n
            }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(COPIES, "title memory by address: no number left for another copy; its pages read zeros")
                });
                return None;
            }
        };
        let offset = shadow.offset_in(start).unwrap_or(0);
        // The offset first: a shader that sees the number sees it too.
        // SAFETY: inside the directory's mapping.
        unsafe {
            entry.add(1).write_volatile(offset as u32);
            entry.write_volatile(n as u32 + 1);
        }
        Some(n)
    }

    /// The pages shaders found empty since the last frame.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(page);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now"
                );
            }
        }
        #[cfg(feature = "diagnostics")]
        if !pages.is_empty() {
            crate::bvhtrace::line(|| {
                let mut sorted = pages.clone();
                sorted.sort_unstable();
                let list: Vec<String> = sorted.iter().map(|&p| format!("{:#x}", u64::from(p) * PAGE)).collect();
                format!("F {} {}", sorted.len(), list.join(" "))
            });
        }
        for page in pages {
            self.map(u64::from(page) * PAGE, "fault");
        }
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it (their indices in `copies`).
    fn name(&mut self, address: u64, why: &str) -> [Option<usize>; 2] {
        [
            self.map(address, why),
            self.map((address & !(CELL - 1)) + CELL, why),
        ]
    }
}

/// A key the list is asked about ([`HostList::asked`]) once a shader of
/// it may store to title memory through an address.
pub(crate) const STORES: usize = usize::MAX - 0x6f;

/// Bytes of a block of list upload memory looked through for title
/// addresses, from a root SRV's address on: some 340 geometry descriptions.
const SCAN: u64 = 16 << 10;

/// The 8-byte values of `bytes` (from an 8-byte boundary) that are title
/// addresses as the builder writes them: bits 32 to 39 set, nothing above,
/// and a low dword of 64 KiB or more (as for an address in two constants;
/// an address that is not the title's memory is refused when named).
fn title_addresses(bytes: &[u8]) -> impl Iterator<Item = u64> + '_ {
    bytes.chunks_exact(8).filter_map(|q| {
        let v = u64::from_le_bytes(q.try_into().ok()?);
        ((1..=0xFF).contains(&(v >> 32)) && v as u32 >= 0x1_0000).then_some(v)
    })
}

/// `VICTORY_MEMORY_ALL=1`: every list that runs a shader reaching title
/// memory by address brings every numbered copy up to date and, if the
/// shader can store through an address, declares every one written, as
/// before (see [`ready`]).
fn all_copies() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MEMORY_ALL").is_some_and(|v| v == "1"))
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module; `records`: the root
/// views of the shader's root SRVs, whose blocks of list upload memory may
/// hold title addresses), every numbered copy has its view in `heap`'s run
/// and is brought up to date when `list` is submitted (and, with `writes`,
/// copied back after). None if the host cannot hold it; the draw is then
/// dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    records: &[u64],
    writes: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.take_faults();
    }
    let numbered = m.copies.len();
    // The copies the dispatch names, and those of list upload memory.
    let mut named = Vec::new();
    for &address in addresses {
        named.extend(m.name(address, "root view").into_iter().flatten());
    }
    // List upload memory (`crate::list::upload`): the builder reaches it
    // through pointers it reads out of its root views. A chunk is a
    // window; once numbered it stays so.
    let mut upload = Vec::new();
    if let Some((start, len)) = crate::list::upload::span() {
        for chunk in (start..start + len).step_by(crate::list::upload::CHUNK) {
            upload.extend(m.map(chunk, "upload memory"));
        }
    }
    // Title addresses in the blocks of list upload memory a dispatch reads
    // through its root SRVs: the builder's leaf rehydration takes its
    // geometry (index and vertex buffers) by address out of its 48-byte
    // records there, which no root argument names. In rt-int1, 638 of the
    // 2,139 such addresses lay in no copy when the stage ran: those leaves
    // read zeros. The builder fills the records before it records the
    // dispatch. Looked through up to what the list has handed out (the
    // block, and those after it), [`SCAN`] bytes at most.
    for &address in records {
        let Some(handed) = crate::list::upload::handed_end(address) else {
            continue;
        };
        let from = address & !7;
        let end = handed.min(address.saturating_add(SCAN));
        if end <= from {
            continue;
        }
        let mut bytes = vec![0u8; (end - from) as usize];
        if crate::host::read_into(from as *const u8, &mut bytes) {
            for pointer in title_addresses(&bytes) {
                named.extend(m.name(pointer, "address in upload memory").into_iter().flatten());
            }
        }
    }
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units. A constant that only looks like one names
    // memory nothing reads; one that is not the title's memory is refused.
    // The low dword of a pair is not taken in 256-byte units as well: of
    // the windows the builder's dispatches named that way, 717 of 802 in
    // its first frame were such low dwords (run rt-trace8, which ran out of
    // numbers in that frame), 2,144 of 2,256 over a run (rt-dry1).
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            named.extend(m.name(u64::from(c) | u64::from(high) << 32, "constant pair").into_iter().flatten());
        } else if c >= 0x0100_0000 {
            named.extend(m.name(u64::from(c) << 8, "constant in 256 bytes").into_iter().flatten());
        }
    }
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made) = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            // Held for the heap's life: not given back with a list's slots.
            let base = heap.dynamic(COPIES + 1)?;
            heap.null_raw_uav_at(base);
            (base, 0)
        }
    };
    for (n, shadow) in m.copies.iter().enumerate().skip(made as usize) {
        let (_, len) = shadow.span();
        heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, len);
    }
    m.heaps.insert(key, (base, m.copies.len() as u32));
    const READ: usize = usize::MAX - 0x6d;
    const WRITE: usize = usize::MAX - 0x6e;
    let first_read = list.first_use(READ);
    if writes {
        list.first_use(STORES);
    }
    if all_copies() {
        // Once per list, and for copies numbered since.
        let first_write = writes && list.first_use(WRITE);
        for (n, shadow) in m.copies.iter().enumerate() {
            let (start, len) = shadow.span();
            if first_read || n >= numbered {
                shadow.refresh(list, start, len);
            }
            if writes && (first_write || n >= numbered) {
                shadow.gpu_writes(list, start, len);
            }
        }
    } else {
        // Brought up to date at the list's submission: a copy just
        // numbered, and each copy the list's dispatches name and the list
        // upload memory's, once per list; declared written, with `writes`,
        // the first two. All numbered copies for every list cost the
        // builder's steady frames (360 dispatches naming 100 windows) a
        // compare of the 1,023 numbered windows, 4 GiB, after each list:
        // with its dispatches bound and not run, a median of 975 ms a frame
        // (run rt-dry1) against 353 for the scene without the builder
        // (rt-off1, the machine shared with other runs' builds).
        // shortcut: memory a stage reaches only through pointers, in a copy
        // no dispatch names, is brought up to date once, when numbered: the
        // title's writes to it after that are not seen; and the stage's
        // writes there are not copied back into title memory (shaders read
        // them from the copy; a page the title writes later is copied over
        // them).
        // (Per list and copy: one key for the refresh, one for the writes.)
        const NAMED: usize = usize::MAX - (1 << 32);
        for n in (numbered..m.copies.len()).chain(named).chain(upload.iter().copied()) {
            let Some(shadow) = m.copies.get(n) else { continue };
            let (start, len) = shadow.span();
            if list.first_use(NAMED + 2 * n) {
                shadow.refresh(list, start, len);
            }
            if writes && !upload.contains(&n) && list.first_use(NAMED + 2 * n + 1) {
                shadow.gpu_writes(list, start, len);
            }
        }
    }
    if first_read {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        faults: m.faults.2,
        base,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn title_addresses_are_found_in_geometry_records() {
        // A geometry description as the builder copies it into upload memory
        // (48 bytes): no transform, index and vertex formats, counts, the
        // index buffer, the vertex buffer and its stride.
        let mut record = Vec::new();
        for v in [0u64, 0x10_0000_0035, 0x508_0000_1263, 0xb_13f6_53a0, 0xb_13f0_8bc0, 8] {
            record.extend_from_slice(&v.to_le_bytes());
        }
        let found: Vec<u64> = super::title_addresses(&record).collect();
        assert_eq!(found, [0xb_13f6_53a0, 0xb_13f0_8bc0]);
    }
}
