use std::future::poll_fn;
use std::io;
use std::ops::Range;
use std::task::Poll;

use crate::directory::{FileSlice, OwnedBytes};

/// Physical batch transport limits used to conservatively admit selective reads.
///
/// Supply the same limits as the directory's batch coalescer. These are cold-read
/// upper bounds conditional on these transport limits and no cache hits;
/// they are not exact request counts after cache filtering. Whole reads use the ordinary async
/// file read and are not split by these batch limits.
#[derive(Clone, Copy, Debug)]
pub struct CandidateReadOptions {
    /// Largest gap the batch transport may fetch between requested ranges.
    pub max_merge_gap: usize,
    /// Maximum body size of a physical batch request. Must be nonzero.
    pub max_read_bytes: usize,
    /// Minimum cold bytes to save for each extra request after reading a position directory.
    pub min_saving_per_extra_get: usize,
}

impl Default for CandidateReadOptions {
    fn default() -> Self {
        Self {
            max_merge_gap: 0,
            max_read_bytes: usize::MAX,
            min_saving_per_extra_get: 0,
        }
    }
}

/// An inspectable cold-read alternative, chosen before paying metadata I/O.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CandidateReadPlan {
    pub(crate) whole: bool,
    pub(crate) bytes_upper_bound: usize,
    pub(crate) gets_upper_bound: usize,
    pub(crate) dependent_phases: usize,
}

impl CandidateReadPlan {
    pub(crate) fn whole(len: usize) -> Self {
        Self {
            whole: true,
            bytes_upper_bound: len,
            gets_upper_bound: usize::from(len != 0),
            dependent_phases: usize::from(len != 0),
        }
    }

    pub(crate) fn choose(
        len: usize,
        prefix: usize,
        payload: usize,
        ranges: usize,
        gaps: usize,
        options: CandidateReadOptions,
    ) -> Self {
        let bytes = prefix.saturating_add(payload).saturating_add(gaps);
        if bytes >= len {
            return Self::whole(len);
        }
        let cap = options.max_read_bytes.max(1);
        // Each range can add one partial body; splitting adds at most total/cap.
        let gets = prefix
            .div_ceil(cap)
            .saturating_add(ranges)
            .saturating_add(payload.saturating_add(gaps) / cap);
        Self {
            whole: false,
            bytes_upper_bound: bytes,
            gets_upper_bound: gets,
            dependent_phases: usize::from(prefix != 0) + usize::from(ranges != 0),
        }
    }

    pub(crate) fn record(self, kind: &str) {
        debug!("Candidate {kind} read plan: whole={}, cold_bytes_upper={}, cold_gets_upper={}, dependent_phases={}",
            self.whole, self.bytes_upper_bound, self.gets_upper_bound, self.dependent_phases);
    }
}

/// Preserve the directory's normal whole-range transport without batch splitting.
pub(crate) async fn read_whole(file: &FileSlice) -> io::Result<OwnedBytes> {
    use crate::HasLen;
    if file.len() == 0 {
        return Ok(OwnedBytes::empty());
    }
    let bytes = match file.read_bytes_async().await {
        Ok(bytes) => bytes,
        // Older adapters may implement batch async alone. Preserve compatibility
        // without hiding real I/O failures or changing normal whole-read transport.
        Err(error) if error.kind() == io::ErrorKind::Unsupported => {
            return read_exact(file, 0..file.len()).await;
        }
        Err(error) => return Err(error),
    };
    check_length(&bytes, file.len())?;
    Ok(bytes)
}

pub(crate) fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn check_length(bytes: &OwnedBytes, expected: usize) -> io::Result<()> {
    if bytes.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "unexpected byte count reading inverted index block",
        ));
    }
    Ok(())
}

pub(crate) async fn read_exact(file: &FileSlice, range: Range<usize>) -> io::Result<OwnedBytes> {
    if range.is_empty() {
        return Ok(OwnedBytes::empty());
    }
    let expected = range.len();
    // Use the bounded batch transport even for a single logical leader or metadata range.
    let mut response = file.read_bytes_batch_async(vec![range]).await?;
    if response.len() != 1 {
        return Err(invalid_data(
            "invalid inverted index singleton batch response",
        ));
    }
    let bytes = response.pop().unwrap();
    check_length(&bytes, expected)?;
    Ok(bytes)
}

/// Give the executor a scheduling point even if every read was served from cache.
pub(crate) async fn yield_once() {
    let mut yielded = false;
    poll_fn(|cx| {
        if yielded {
            return Poll::Ready(());
        }
        yielded = true;
        cx.waker().wake_by_ref();
        Poll::Pending
    })
    .await;
}

/// Read logical block ranges in input order, reusing bytes already fetched with the metadata.
pub(crate) async fn read_blocks(
    file: &FileSlice,
    prefix: &OwnedBytes,
    ranges: &[Range<usize>],
) -> io::Result<Vec<OwnedBytes>> {
    Ok(read_blocks_with_optional_tail(file, prefix, ranges, None)
        .await?
        .0)
}

/// An optional tail is a hint to the physical planner, never a required read.
pub(crate) async fn read_blocks_with_optional_tail(
    file: &FileSlice,
    prefix: &OwnedBytes,
    ranges: &[Range<usize>],
    tail: Option<Range<usize>>,
) -> io::Result<(Vec<OwnedBytes>, Option<OwnedBytes>)> {
    let remote_ranges: Vec<_> = ranges
        .iter()
        .filter(|range| !range.is_empty() && range.end > prefix.len())
        .map(|range| range.start.max(prefix.len())..range.end)
        .collect();
    let (remote, tail_bytes) = match tail {
        Some(tail) if tail.end <= prefix.len() => (
            if remote_ranges.is_empty() {
                Vec::new()
            } else {
                file.read_bytes_batch_async(remote_ranges).await?
            },
            Some(prefix.slice(tail)),
        ),
        Some(tail) if !remote_ranges.is_empty() && tail.start >= prefix.len() => {
            let len = tail.len();
            let (blocks, tail) = file
                .read_bytes_batch_with_optional_tail_async(remote_ranges, tail)
                .await?;
            if let Some(bytes) = &tail {
                check_length(bytes, len)?;
            }
            (blocks, tail)
        }
        _ if remote_ranges.is_empty() => (Vec::new(), None),
        _ => (file.read_bytes_batch_async(remote_ranges).await?, None),
    };
    let mut remote_bytes = remote.into_iter();

    let result = ranges
        .iter()
        .map(|range| {
            if range.is_empty() {
                return Ok(OwnedBytes::empty());
            }
            if range.end <= prefix.len() {
                return Ok(prefix.slice(range.clone()));
            }
            let remote_start = range.start.max(prefix.len());
            let remote = remote_bytes
                .next()
                .ok_or_else(|| invalid_data("missing inverted index block from batch read"))?;
            check_length(&remote, range.end - remote_start)?;
            if range.start == remote_start {
                return Ok(remote);
            }

            let mut combined = Vec::with_capacity(range.len());
            combined.extend_from_slice(&prefix[range.start..remote_start]);
            combined.extend_from_slice(&remote);
            Ok(OwnedBytes::new(combined))
        })
        .collect::<io::Result<Vec<_>>>()?;
    if remote_bytes.next().is_some() {
        return Err(invalid_data(
            "unexpected extra inverted index block from batch read",
        ));
    }
    Ok((result, tail_bytes))
}

/// Decode the same stop-bit VInt format as the legacy readers, with a length and overflow check.
pub(crate) fn read_vint(bytes: &mut &[u8], max_bits: u32) -> io::Result<u64> {
    let mut value = 0;
    for shift in (0..max_bits).step_by(7) {
        let (&byte, remaining) = bytes
            .split_first()
            .ok_or_else(|| invalid_data("truncated inverted index variable integer"))?;
        *bytes = remaining;
        let part = u64::from(byte & 0x7f);
        if part >= (1 << (max_bits - shift).min(7)) {
            return Err(invalid_data("inverted index variable integer overflow"));
        }
        value |= part << shift;
        if byte & 0x80 != 0 {
            return Ok(value);
        }
    }
    Err(invalid_data("unterminated inverted index variable integer"))
}
