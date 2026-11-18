use std::io;
use std::ops::Range;

use common::VInt;

use crate::directory::{FileSlice, OwnedBytes};
use crate::postings::block_read::{
    invalid_data, read_blocks, read_exact, read_vint, read_whole, CandidateReadOptions,
    CandidateReadPlan,
};
use crate::postings::compression::{compressed_block_size, BlockDecoder};
use crate::HasLen;

use super::COMPRESSION_BLOCK_SIZE;

// Fetch the bit widths and some payload in one operation for small position lists.
const POSITION_PREFIX_BYTES: usize = 4 * 1024;

struct PositionLayout {
    // Byte offset of each compressed block, including the variable-integer tail.
    offsets: Vec<usize>,
    // Cached payload starts at offsets[0]. The directory may have needed a supplementary read.
    payload_prefix: OwnedBytes,
}

/// Reads position blocks only for the posting offsets retained in the candidate batch.
pub(crate) struct CandidatePositionReader {
    file: FileSlice,
    layout: Option<PositionLayout>,
    prefetched: Vec<(usize, OwnedBytes)>,
    options: CandidateReadOptions,
}

impl CandidatePositionReader {
    pub(crate) fn new(file: FileSlice) -> Self {
        Self {
            file,
            layout: None,
            prefetched: Vec::new(),
            options: CandidateReadOptions::default(),
        }
    }

    pub(crate) fn set_read_options(&mut self, options: CandidateReadOptions) {
        self.options = options;
    }

    pub(crate) fn read_upper_bound(
        &self,
        ranges: &[Range<u64>],
        total: Option<u64>,
    ) -> io::Result<usize> {
        match total {
            Some(total) => Ok(self.plan(ranges, total)?.bytes_upper_bound),
            None => Ok(self.file.len()), // Reserve for a possible whole read before probing.
        }
    }

    fn plan(&self, ranges: &[Range<u64>], total: u64) -> io::Result<CandidateReadPlan> {
        if ranges.is_empty() {
            return Ok(CandidateReadPlan::whole(0));
        }
        let mut previous_end = 0;
        let mut last_block: Option<u64> = None;
        let mut selected = 0usize;
        let mut gaps = 0usize;
        for range in ranges {
            if range.start < previous_end || range.start >= range.end || range.end > total {
                return Err(invalid_data("invalid position range for posting"));
            }
            let first = range.start / COMPRESSION_BLOCK_SIZE as u64;
            let last = (range.end - 1) / COMPRESSION_BLOCK_SIZE as u64;
            let start = last_block.map_or(first, |prev| first.max(prev + 1));
            if start <= last {
                selected = selected
                    .saturating_add(usize::try_from(last - start + 1).unwrap_or(usize::MAX));
                if last_block.is_none_or(|prev| first > prev + 1) {
                    if let Some(prev) = last_block {
                        let omitted = usize::try_from(first - prev - 1).unwrap_or(usize::MAX);
                        gaps = gaps.saturating_add(
                            omitted
                                .saturating_mul(COMPRESSION_BLOCK_SIZE * 4)
                                .min(self.options.max_merge_gap),
                        );
                    }
                }
                last_block = Some(last);
            }
            previous_end = range.end;
        }
        let full_blocks = usize::try_from(total / COMPRESSION_BLOCK_SIZE as u64)
            .map_err(|_| invalid_data("position block count overflow"))?;
        let header =
            full_blocks.saturating_add(VInt(full_blocks as u64).serialize_into(&mut [0; 10]));
        let prefix = self.file.len().min(POSITION_PREFIX_BYTES.max(header));
        let tail = usize::from(last_block == Some(full_blocks as u64));
        let payload = selected
            .saturating_sub(tail)
            .saturating_mul(COMPRESSION_BLOCK_SIZE * 4)
            .saturating_add(tail * (total as usize % COMPRESSION_BLOCK_SIZE) * 5);
        // The transport receives one logical range per block. Even contiguous
        // blocks can fill different capped bodies, so count each as an upper bound.
        Ok(CandidateReadPlan::choose(
            self.file.len(),
            prefix,
            payload,
            selected,
            gaps,
            self.options,
        ))
    }

    async fn load(&mut self, total_positions: Option<u64>, whole: bool) -> io::Result<()> {
        if self.layout.is_some() {
            return Ok(());
        }
        let expected_blocks = total_positions
            .map(|total| {
                usize::try_from(total / COMPRESSION_BLOCK_SIZE as u64)
                    .map_err(|_| invalid_data("position block count overflow"))
            })
            .transpose()?;
        let expected_header = if let Some(blocks) = expected_blocks {
            blocks
                .checked_add(VInt(blocks as u64).serialize_into(&mut [0; 10]))
                .ok_or_else(|| invalid_data("position header length overflow"))?
        } else {
            0
        };
        let prefix_len = self
            .file
            .len()
            .min(POSITION_PREFIX_BYTES.max(expected_header));
        let prefix = if whole {
            read_whole(&self.file).await?
        } else {
            read_exact(&self.file, 0..prefix_len).await?
        };
        let mut bytes = prefix.as_slice();
        let full_blocks = usize::try_from(read_vint(&mut bytes, 64)?)
            .map_err(|_| invalid_data("position block count overflow"))?;
        if expected_blocks.is_some_and(|expected| full_blocks != expected) {
            return Err(invalid_data("position block count disagrees with postings"));
        }
        let header_len = prefix.len() - bytes.len();
        let body_offset = header_len
            .checked_add(full_blocks)
            .filter(|&len| len <= self.file.len())
            .ok_or_else(|| invalid_data("truncated position bit widths"))?;
        let widths_suffix =
            read_exact(&self.file, prefix.len().min(body_offset)..body_offset).await?;
        let widths = bytes.iter().take(full_blocks).chain(widths_suffix.iter());
        let mut offsets = Vec::with_capacity(full_blocks + 1);
        let mut offset = body_offset;
        for &width in widths {
            if width > 32 {
                return Err(invalid_data("invalid position bit width"));
            }
            offsets.push(offset);
            offset = offset
                .checked_add(compressed_block_size(width))
                .filter(|&end| end <= self.file.len())
                .ok_or_else(|| invalid_data("position block exceeds term range"))?;
        }
        let tail_bytes = self.file.len() - offset;
        let (min_tail, max_tail) = if let Some(total) = total_positions {
            let tail_len = (total % COMPRESSION_BLOCK_SIZE as u64) as usize;
            (tail_len, tail_len * 5)
        } else {
            (0, (COMPRESSION_BLOCK_SIZE - 1) * 5)
        };
        if !(min_tail..=max_tail).contains(&tail_bytes) {
            return Err(invalid_data("invalid encoded position tail size"));
        }
        offsets.push(offset);
        let payload_prefix = prefix.slice(body_offset.min(prefix.len())..prefix.len());
        self.layout = Some(PositionLayout {
            offsets,
            payload_prefix,
        });
        Ok(())
    }

    fn select_blocks(
        &self,
        ranges: &[Range<u64>],
        total_positions: Option<u64>,
    ) -> io::Result<Vec<usize>> {
        let layout = self.layout.as_ref().unwrap();
        let tail_block = layout.offsets.len() - 1;
        let tail_len = self.file.len() - layout.offsets[tail_block];
        let tail_start = tail_block as u64 * COMPRESSION_BLOCK_SIZE as u64;
        let available_positions = tail_start + tail_len.min(COMPRESSION_BLOCK_SIZE - 1) as u64;
        let upper_position =
            total_positions.map_or(available_positions, |total| total.min(available_positions));

        let mut block_ids = Vec::new();
        let mut previous_end = 0;
        for range in ranges {
            if range.start < previous_end || range.start >= range.end || range.end > upper_position
            {
                return Err(invalid_data("invalid position range for posting"));
            }
            let first = block_id(range.start);
            let last = block_id(range.end - 1);
            let start = block_ids.last().map_or(first, |&prev| first.max(prev + 1));
            block_ids.extend(start..=last);
            previous_end = range.end;
        }
        Ok(block_ids)
    }

    async fn read_blocks(&self, blocks: &[usize]) -> io::Result<Vec<OwnedBytes>> {
        let layout = self.layout.as_ref().unwrap();
        let mut missing_ranges = Vec::new();
        let bytes: Vec<_> = blocks
            .iter()
            .map(|&id| {
                if let Ok(idx) = self.prefetched.binary_search_by_key(&id, |(id, _)| *id) {
                    return Some(self.prefetched[idx].1.clone());
                }
                let end = layout
                    .offsets
                    .get(id + 1)
                    .copied()
                    .unwrap_or(self.file.len());
                missing_ranges
                    .push(layout.offsets[id] - layout.offsets[0]..end - layout.offsets[0]);
                None
            })
            .collect();
        let mut fetched = read_blocks(
            &self.file.slice_from(layout.offsets[0]),
            &layout.payload_prefix,
            &missing_ranges,
        )
        .await?
        .into_iter();
        // The block-read helper validates that each requested range returned exactly one result.
        Ok(bytes
            .into_iter()
            .map(|bytes| bytes.unwrap_or_else(|| fetched.next().unwrap()))
            .collect())
    }

    async fn prepare_layout(
        &mut self,
        ranges: &[Range<u64>],
        total: Option<u64>,
    ) -> io::Result<()> {
        if self.layout.is_some() {
            return Ok(());
        }
        if let Some(total) = total {
            let plan = self.plan(ranges, total)?;
            plan.record("positions");
            return self.load(Some(total), plan.whole).await;
        }
        // A short term fits in the metadata probe itself. Larger unknown terms describe
        // their compressed block count and widths at the beginning of the .pos slice.
        self.load(None, self.file.len() <= POSITION_PREFIX_BYTES)
            .await?;
        let blocks = self.select_blocks(ranges, None)?;
        let layout = self.layout.as_mut().unwrap();
        let body_start = layout.offsets[0];
        let missing_start = body_start + layout.payload_prefix.len();
        let whole_bytes = self.file.len() - missing_start;
        if whole_bytes == 0 {
            return Ok(());
        }
        let selected = blocks
            .iter()
            .map(|&id| {
                layout.offsets[id].max(missing_start)
                    ..layout
                        .offsets
                        .get(id + 1)
                        .copied()
                        .unwrap_or(self.file.len())
            })
            .filter(|r| r.start < r.end);
        let (bytes, gets) = cold_payload_cost(selected, self.options);
        let extra_gets = gets.saturating_sub(1);
        if gets > 0
            && (bytes >= whole_bytes
                || whole_bytes - bytes
                    < extra_gets.saturating_mul(self.options.min_saving_per_extra_get))
        {
            let remaining = read_whole(&self.file.slice_from(missing_start)).await?;
            if layout.payload_prefix.is_empty() {
                layout.payload_prefix = remaining;
            } else {
                let mut payload = Vec::with_capacity(self.file.len() - body_start);
                payload.extend_from_slice(&layout.payload_prefix);
                payload.extend_from_slice(&remaining);
                layout.payload_prefix = OwnedBytes::new(payload);
            }
        }
        Ok(())
    }

    pub(crate) async fn prefetch(
        &mut self,
        ranges: &[Range<u64>],
        total_positions: Option<u64>,
    ) -> io::Result<()> {
        if !ranges.is_empty() {
            self.prepare_layout(ranges, total_positions).await?;
            let blocks = self.select_blocks(ranges, total_positions)?;
            let bytes = self.read_blocks(&blocks).await?;
            self.prefetched = blocks.into_iter().zip(bytes).collect();
        }
        Ok(())
    }

    pub(crate) async fn read_positions(
        &mut self,
        ranges: &[Range<u64>],
        total_positions: Option<u64>,
    ) -> io::Result<Vec<Vec<u32>>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        self.prepare_layout(ranges, total_positions).await?;
        let blocks = self.select_blocks(ranges, total_positions)?;
        let bytes = self.read_blocks(&blocks).await?;
        let mut blocks = blocks.into_iter().zip(bytes);
        let tail_block = self.layout.as_ref().unwrap().offsets.len() - 1;
        let tail_start = tail_block as u64 * COMPRESSION_BLOCK_SIZE as u64;
        let mut decoder = BlockDecoder::default();
        let mut current_id = None;
        let mut tail = Vec::new();
        let mut output = Vec::with_capacity(ranges.len());
        for range in ranges {
            let mut offset = range.start;
            let range_len = usize::try_from(range.end - range.start)
                .map_err(|_| invalid_data("position output length overflow"))?;
            let mut positions = Vec::with_capacity(range_len);
            let mut absolute = 0u32;
            while offset < range.end {
                let target_id = block_id(offset);
                if current_id != Some(target_id) {
                    let (id, bytes) = blocks
                        .next()
                        .filter(|(id, _)| *id == target_id)
                        .ok_or_else(|| invalid_data("missing decoded position block"))?;
                    current_id = Some(id);
                    if id == tail_block {
                        let mut bytes = bytes.as_slice();
                        while !bytes.is_empty() {
                            tail.push(read_vint(&mut bytes, 32)? as u32);
                            if tail.len() >= COMPRESSION_BLOCK_SIZE {
                                return Err(invalid_data("invalid variable-integer position tail"));
                            }
                        }
                        if total_positions
                            .is_some_and(|total| tail_start + tail.len() as u64 != total)
                        {
                            return Err(invalid_data(
                                "position tail length disagrees with postings",
                            ));
                        }
                    } else {
                        let width = (bytes.len() * 8 / COMPRESSION_BLOCK_SIZE) as u8;
                        decoder.uncompress_block_unsorted(&bytes, width, false);
                    }
                }
                let deltas = if target_id == tail_block {
                    &tail
                } else {
                    decoder.output_array()
                };
                let start = (offset % COMPRESSION_BLOCK_SIZE as u64) as usize;
                let len =
                    (range.end - offset).min((COMPRESSION_BLOCK_SIZE - start) as u64) as usize;
                let deltas = deltas
                    .get(start..start + len)
                    .ok_or_else(|| invalid_data("posting extends past the position tail"))?;
                for &delta in deltas {
                    absolute = absolute
                        .checked_add(delta)
                        .ok_or_else(|| invalid_data("invalid position delta"))?;
                    positions.push(absolute);
                }
                offset += len as u64;
            }
            output.push(positions);
        }
        Ok(output)
    }
}

// The block ranges are sorted and disjoint. Model the configured cold batch transport,
// including over-read gaps and maximum request size. Cached payload is excluded by the caller.
fn cold_payload_cost(
    ranges: impl Iterator<Item = Range<usize>>,
    options: CandidateReadOptions,
) -> (usize, usize) {
    let cap = options.max_read_bytes.max(1);
    let mut last: Option<Range<usize>> = None;
    let (mut bytes, mut gets) = (0usize, 0usize);
    for range in ranges {
        let mut start = range.start;
        if let Some(previous) = &mut last {
            let end = previous.start.saturating_add(cap).min(range.end);
            if start <= previous.end.saturating_add(options.max_merge_gap) && end > start {
                bytes += end - previous.end;
                previous.end = end;
                start = end;
            }
        }
        if start < range.end {
            let len = range.end - start;
            bytes += len;
            gets += len.div_ceil(cap);
            let last_start = start + (len - 1) / cap * cap;
            last = Some(last_start..range.end);
        }
    }
    (bytes, gets)
}

fn block_id(position: u64) -> usize {
    (position / COMPRESSION_BLOCK_SIZE as u64) as usize
}
