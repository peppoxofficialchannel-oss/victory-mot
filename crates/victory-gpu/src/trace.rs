//! `VICTORY_TRACE=<file.json>`: a timeline of the pipeline's stages in the
//! Chrome Trace format, which `ui.perfetto.dev` opens.
//!
//! Every stage of a frame is a complete interval (`"ph":"X"`) with a name,
//! the thread that ran it and a `ts`/`dur` in microseconds from the first
//! span of the process. With one row per thread the critical path and the
//! gaps between stages can be seen.
//!
//! Cost when it is off: one relaxed load per [`span`] (the switch is read
//! once, in a [`OnceLock`]), and nothing else. A span is a clock reading
//! and a push into a thread-local `Vec`; there is no global mutex per span
//! and no formatting per span. The thread-local buffer hands its records to
//! the shared buffer a few thousand at a time, and the JSON is written only
//! on a flush.
//!
//! Flush: a background thread of ours writes the whole timeline to the file
//! every five seconds, replacing it whole, and [`flush`] is called where the
//! title's process exits cleanly (`kernel::exit`). The periodic writer is
//! there because the title may end the process in ways we do not see
//! (`TerminateProcess` from another process, a crash): the file then holds
//! everything up to the last write, five seconds at most behind. Every write
//! logs one line — `tracing: N spans, M threads, file X` — so a run's log
//! says at once whether the file has been written at all ([`trace_written`]
//! counts them).

use std::cell::RefCell;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// One finished interval, as the JSON needs it.
struct Record {
    name: &'static str,
    /// Microseconds from the process's first span.
    ts: u64,
    dur: u64,
    /// The Windows thread id that ran the interval.
    tid: u32,
}

/// The file `VICTORY_TRACE` names, read once.
fn configured() -> Option<&'static std::path::Path> {
    static PATH: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| {
        let path = std::env::var_os("VICTORY_TRACE")?;
        if path.is_empty() {
            return None;
        }
        // The timeline is written by a thread of ours every few seconds, so
        // the file exists long before the run ends. The epoch is taken now,
        // before the first span, so the first `ts` is meaningful.
        start_writer();
        let _ = micros();
        Some(std::path::PathBuf::from(path))
    })
    .as_deref()
}

/// Whether tracing is on this run.
pub fn enabled() -> bool {
    configured().is_some()
}

/// The file this run traces to, if any.
pub fn configured_path() -> Option<&'static std::path::Path> {
    configured()
}

/// Microseconds from the process's first span (`Instant::now` once, never
/// again per span).
fn micros() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_micros() as u64
}

fn thread_id() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

/// The most intervals a thread-local buffer holds before handing them over.
const HAND_OVER: usize = 4096;

/// Every interval of the process so far, the buffer the flushes read.
static ALL: Mutex<Vec<Record>> = Mutex::new(Vec::new());

thread_local! {
    /// This thread's intervals not yet handed to [`ALL`].
    static BUF: Buf = const { Buf { records: RefCell::new(Vec::new()) } };
}

struct Buf {
    records: RefCell<Vec<Record>>,
}

impl Drop for Buf {
    fn drop(&mut self) {
        // The thread ends: its intervals go to the shared buffer, so a
        // flush after it is gone still has them.
        let taken = std::mem::take(&mut *self.records.borrow_mut());
        hand_over(taken);
    }
}

/// Appends `records` to [`ALL`], a lock taken once per batch, not per span.
fn hand_over(records: Vec<Record>) {
    if records.is_empty() {
        return;
    }
    ALL.lock().unwrap_or_else(|e| e.into_inner()).extend(records);
}

/// An interval being timed: its name, its start, and its `ts` in
/// microseconds from the epoch. Dropped at the end of the stage; nothing at
/// all happens when tracing is off.
pub struct Span(Option<(&'static str, Instant, u64)>);

/// Starts an interval named `name`, or a do-nothing one when tracing is off.
#[inline]
pub fn span(name: &'static str) -> Span {
    if enabled() {
        Span(Some((name, Instant::now(), micros())))
    } else {
        Span(None)
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let Some((name, started, ts)) = self.0.as_ref() else {
            return;
        };
        let record = Record {
            name,
            ts: *ts,
            dur: started.elapsed().as_micros() as u64,
            tid: thread_id(),
        };
        let taken = BUF.with(|buf| {
            let mut records = buf.records.borrow_mut();
            records.push(record);
            if records.len() >= HAND_OVER {
                Some(std::mem::take(&mut *records))
            } else {
                None
            }
        });
        if let Some(taken) = taken {
            hand_over(taken);
        }
    }
}

/// Writes the whole timeline to the configured file, if tracing is on. The
/// JSON is built from [`ALL`] under the lock, then written outside it.
///
/// Every write says so in one line of the log (`tracing: N spans, M threads,
/// file X`, [`trace_written`]): a run whose log has none of those lines has
/// written nothing, and that is the first thing to look at. The file is
/// replaced whole (written beside, then renamed over): a reader — a person,
/// a viewer watching it grow — never catches it truncated to zero between
/// one write and the next.
pub fn flush() {
    let Some(path) = configured() else {
        return;
    };
    let (text, spans, threads) = {
        let records = ALL.lock().unwrap_or_else(|e| e.into_inner());
        let mut text = String::with_capacity(64 + records.len() * 88);
        text.push_str("{\"displayTimeUnit\":\"ms\",\"traceEvents\":[");
        let pid = std::process::id();
        let mut tids: Vec<u32> = Vec::new();
        for (i, record) in records.iter().enumerate() {
            if i > 0 {
                text.push(',');
            }
            if !tids.contains(&record.tid) {
                tids.push(record.tid);
            }
            let _ = write!(
                text,
                "{{\"pid\":{pid},\"tid\":{},\"ph\":\"X\",\"name\":\"{}\",\"ts\":{},\"dur\":{}}}",
                record.tid, record.name, record.ts, record.dur
            );
        }
        text.push_str("]}");
        (text, records.len(), tids.len())
    };
    match write_whole(path, text.as_bytes()) {
        Ok(()) => {
            WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::info!("tracing: {spans} spans, {threads} threads, file {}", path.display());
        }
        Err(error) => tracing::warn!(
            error = %error,
            "tracing: the timeline was not written to {} ({spans} spans, {threads} threads)",
            path.display()
        ),
    }
}

/// How many times the timeline has been written to the file this run.
static WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`WRITES`] as it stands.
pub fn trace_written() -> u64 {
    WRITES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Writes `bytes` to `path` by replacing the file whole: they go to a file
/// beside it, which is then renamed over it. A reader never sees a
/// half-written (or, after `File::create`, momentarily empty) timeline. If
/// the rename is refused — a reader holds the file open — the bytes are
/// written in place instead, truncated for a moment: a file that grows late
/// is better than one that does not grow.
fn write_whole(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut beside = path.as_os_str().to_owned();
    beside.push(".new");
    let beside = PathBuf::from(beside);
    std::fs::write(&beside, bytes)?;
    match std::fs::rename(&beside, path) {
        Ok(()) => Ok(()),
        Err(_) => {
            let _ = std::fs::remove_file(&beside);
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .open(path)?;
            file.write_all(bytes)?;
            file.set_len(bytes.len() as u64)
        }
    }
}

/// The thread that overwrites the file every five seconds, started once when
/// tracing is on. This, not the exit path, is what makes the file reliable
/// when the process dies without our being told.
///
/// A thread that could not be made, or a write that panicked, says so in the
/// log: the failure that stopped a run's timeline from growing (run trace3,
/// three minutes with a 0 KB file) was silent, and a silent one costs the
/// whole run.
fn start_writer() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new()
            .name("victory-trace".into())
            .spawn(|| loop {
                std::thread::sleep(Duration::from_secs(5));
                let wrote = std::panic::catch_unwind(std::panic::AssertUnwindSafe(flush));
                if wrote.is_err() {
                    tracing::warn!("tracing: a write of the timeline panicked; the next one follows");
                }
            });
        if let Err(error) = spawned {
            tracing::warn!(
                error = %error,
                "tracing: no thread for the timeline: the file is written only at exit (VICTORY_TRACE)"
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracing_off_writes_no_file() {
        // The switch is read once per process: leave the test out if the
        // harness happens to run with it set.
        if std::env::var_os("VICTORY_TRACE").is_some() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let candidate = dir.path().join("trace.json");
        // With the switch unset there is no configured path, and the file
        // the tracer writes is only ever that path.
        assert!(!enabled());
        assert!(configured_path().is_none());
        // A span and a flush must leave nothing behind.
        {
            let _span = span("test");
        }
        flush();
        assert!(!candidate.exists());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}
