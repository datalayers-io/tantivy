use std::future::poll_fn;
use std::io;
use std::ops::Range;
use std::task::Poll;

use crate::directory::{FileSlice, OwnedBytes};

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
    let remote_ranges: Vec<_> = ranges
        .iter()
        .filter(|range| !range.is_empty() && range.end > prefix.len())
        .map(|range| range.start.max(prefix.len())..range.end)
        .collect();
    let mut remote_bytes = if remote_ranges.is_empty() {
        Vec::new()
    } else {
        file.read_bytes_batch_async(remote_ranges).await?
    }
    .into_iter();

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
    Ok(result)
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
