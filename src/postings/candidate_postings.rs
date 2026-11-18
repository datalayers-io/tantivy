use std::io;
use std::ops::Range;

use common::VInt;

use super::block_read::{
    invalid_data, read_blocks_with_optional_tail, read_exact, read_vint, read_whole, yield_once,
    CandidateReadOptions, CandidateReadPlan,
};
use super::compression::{compressed_block_size, BlockDecoder, COMPRESSION_BLOCK_SIZE};
use super::skip::skip_entry_len;
use super::{BlockInfo, SkipReader};
use crate::directory::{FileSlice, OwnedBytes};
use crate::positions::CandidatePositionReader;
use crate::schema::IndexRecordOption;
use crate::{DocId, HasLen, TERMINATED};

// Use one read for small terms that need both a skip table and postings payload.
const SMALL_POSTINGS_BYTES: usize = 4 * 1024;
const DECODE_BLOCKS_PER_YIELD: usize = 128;

/// A posting that matched the document candidates. Positions are materialized separately.
#[derive(Clone, Debug)]
pub(crate) struct CandidatePosting {
    pub(crate) doc: DocId,
    pub(crate) term_freq: u32,
    pub(crate) positions: Vec<u32>,
    position_offset: Option<u64>,
}

struct PostingBlock {
    bytes: Range<usize>,
    last_doc: DocId,
    previous_doc: DocId,
    position_offset: u64,
    info: BlockInfo,
}

struct PostingsLayout {
    blocks: Vec<PostingBlock>,
    record_option: IndexRecordOption,
    total_positions: Option<u64>,
    prefix: OwnedBytes,
}

impl PostingsLayout {
    fn parse(
        doc_freq: u32,
        file_len: usize,
        mut record_option: IndexRecordOption,
        prefix: &OwnedBytes,
    ) -> io::Result<Self> {
        let full_blocks = doc_freq as usize / COMPRESSION_BLOCK_SIZE;
        let (skip_data, body_start) = if full_blocks == 0 {
            (OwnedBytes::empty(), 0)
        } else {
            let mut bytes = prefix.as_slice();
            let skip_len = usize::try_from(read_vint(&mut bytes, 64)?)
                .map_err(|_| invalid_data("postings skip length overflow"))?;
            let actual_len = full_blocks * skip_entry_len(record_option);
            if skip_len == full_blocks * skip_entry_len(IndexRecordOption::Basic) {
                // Numerical JSON terms use Basic encodings even in fields with frequencies.
                record_option = IndexRecordOption::Basic;
            } else if skip_len != actual_len {
                return Err(invalid_data("invalid postings skip table length"));
            }
            let header_len = prefix.len() - bytes.len();
            let body_start = header_len + skip_len;
            if body_start > prefix.len() || body_start > file_len {
                return Err(invalid_data("truncated postings skip table"));
            }
            (prefix.slice(header_len..body_start), body_start)
        };

        let mut skip_reader = SkipReader::new(skip_data, doc_freq, record_option);
        let mut blocks = Vec::with_capacity(doc_freq as usize / COMPRESSION_BLOCK_SIZE + 1);
        let mut offset = body_start;
        while skip_reader.has_remaining_docs() {
            let last_doc = skip_reader.last_doc_in_block();
            let previous_doc = skip_reader.last_doc_in_previous_block;
            let info = skip_reader.block_info();
            let block_end = match info {
                BlockInfo::BitPacked {
                    doc_num_bits,
                    tf_num_bits,
                    tf_sum,
                    ..
                } => {
                    if tf_num_bits > 32
                        || last_doc >= TERMINATED
                        || last_doc
                            < previous_doc
                                + (COMPRESSION_BLOCK_SIZE - 1) as u32
                                + u32::from(!blocks.is_empty())
                        || (record_option.has_positions() && tf_sum < COMPRESSION_BLOCK_SIZE as u32)
                    {
                        return Err(invalid_data("invalid packed postings block metadata"));
                    }
                    offset
                        .checked_add(compressed_block_size(doc_num_bits + tf_num_bits))
                        .ok_or_else(|| invalid_data("postings block byte offset overflow"))?
                }
                BlockInfo::VInt { num_docs } => {
                    if file_len - offset < num_docs as usize {
                        return Err(invalid_data("truncated variable-integer postings block"));
                    }
                    file_len
                }
            };
            if block_end > file_len {
                return Err(invalid_data("postings block exceeds term range"));
            }
            blocks.push(PostingBlock {
                bytes: offset..block_end,
                last_doc,
                previous_doc,
                position_offset: skip_reader.position_offset(),
                info,
            });
            offset = block_end;
            skip_reader.advance();
        }
        if offset != file_len {
            return Err(invalid_data("unexpected bytes after final postings block"));
        }
        Ok(Self {
            blocks,
            record_option,
            total_positions: (record_option.has_positions()
                && (doc_freq as usize).is_multiple_of(COMPRESSION_BLOCK_SIZE))
            .then(|| skip_reader.position_offset()),
            prefix: OwnedBytes::empty(),
        })
    }
}

/// Candidate-driven access to one term. Stores the skip table across candidate batches.
pub(crate) struct CandidatePostingsReader {
    doc_freq: u32,
    postings: FileSlice,
    positions: Option<CandidatePositionReader>,
    record_option: IndexRecordOption,
    requested_option: IndexRecordOption,
    layout: Option<PostingsLayout>,
    options: CandidateReadOptions,
}

impl CandidatePostingsReader {
    pub(crate) fn new(
        doc_freq: u32,
        postings: FileSlice,
        positions: Option<FileSlice>,
        record_option: IndexRecordOption,
        requested_option: IndexRecordOption,
    ) -> Self {
        Self {
            doc_freq,
            postings,
            positions: positions.map(CandidatePositionReader::new),
            record_option,
            requested_option,
            layout: None,
            options: CandidateReadOptions::default(),
        }
    }

    pub(crate) fn set_read_options(&mut self, options: CandidateReadOptions) {
        self.options = options;
        if let Some(positions) = &mut self.positions {
            positions.set_read_options(options);
        }
    }

    fn plan_docs(&self, candidates: usize) -> CandidateReadPlan {
        let full_blocks = self.doc_freq as usize / COMPRESSION_BLOCK_SIZE;
        let tail_docs = self.doc_freq as usize % COMPRESSION_BLOCK_SIZE;
        let selected = candidates.min(full_blocks + usize::from(tail_docs != 0));
        let streams = 1 + usize::from(self.record_option.has_freq());
        let block_bytes = COMPRESSION_BLOCK_SIZE * 4 * streams;
        // Include the tail in addition to all potentially selected full blocks:
        // an overestimate avoids assuming where any candidate lives before the skip read.
        let payload = selected
            .min(full_blocks)
            .saturating_mul(block_bytes)
            .saturating_add(if selected != 0 {
                tail_docs * 5 * streams
            } else {
                0
            });
        let skip_len = full_blocks * skip_entry_len(self.record_option);
        let prefix = if full_blocks == 0 {
            0
        } else {
            skip_len + VInt(skip_len as u64).serialize_into(&mut [0; 10])
        };
        let gaps = selected
            .saturating_sub(1)
            .saturating_mul(self.options.max_merge_gap);
        if self.postings.len() <= SMALL_POSTINGS_BYTES {
            CandidateReadPlan::whole(self.postings.len())
        } else {
            CandidateReadPlan::choose(
                self.postings.len(),
                prefix,
                payload,
                selected,
                gaps,
                self.options,
            )
        }
    }

    async fn load_whole(&mut self) -> io::Result<()> {
        let bytes = read_whole(&self.postings).await?;
        let mut layout = PostingsLayout::parse(
            self.doc_freq,
            self.postings.len(),
            self.record_option,
            &bytes,
        )?;
        layout.prefix = bytes;
        // Skip metadata omits the tail's frequency sum. Decode just that bounded
        // tail so position planning can know the total before reading its header.
        if layout.record_option.has_positions() && self.requested_option.has_freq() {
            if let Some(block) = layout.blocks.last() {
                if matches!(block.info, BlockInfo::VInt { .. }) {
                    layout.total_positions = self.decode_block(
                        &layout,
                        block,
                        &layout.prefix[block.bytes.clone()],
                        Some(&[]),
                        &mut Vec::new(),
                    )?;
                }
            }
        }
        self.layout = Some(layout);
        Ok(())
    }

    /// Prepare the skip table for subsequent postings reads.
    pub(crate) async fn load(&mut self) -> io::Result<()> {
        if self.layout.is_none() {
            let full_blocks = self.doc_freq as usize / COMPRESSION_BLOCK_SIZE;
            let skip_len = full_blocks * skip_entry_len(self.record_option);
            let prefix_len = if full_blocks == 0 {
                0
            } else if self.postings.len() <= SMALL_POSTINGS_BYTES {
                self.postings.len()
            } else {
                // The field's record option gives an upper bound. A Basic term in a JSON field
                // can have a smaller skip table. Keep any bytes read through into the payload.
                skip_len + VInt(skip_len as u64).serialize_into(&mut [0; 10])
            };
            let prefix = read_exact(&self.postings, 0..prefix_len.min(self.postings.len())).await?;
            let mut layout = PostingsLayout::parse(
                self.doc_freq,
                self.postings.len(),
                self.record_option,
                &prefix,
            )?;
            layout.prefix = prefix;
            self.layout = Some(layout);
        }
        Ok(())
    }

    /// Read the driving posting list in one async operation, leaving its positions unloaded.
    pub(crate) async fn all_docs(&mut self) -> io::Result<Vec<CandidatePosting>> {
        if self
            .layout
            .as_ref()
            .is_none_or(|layout| layout.prefix.len() != self.postings.len())
        {
            self.load_whole().await?;
        }
        let layout = self.layout.as_ref().unwrap();
        let bytes = &layout.prefix;
        let mut postings = Vec::with_capacity(self.doc_freq as usize);
        for (block_idx, block) in layout.blocks.iter().enumerate() {
            if block_idx != 0 && block_idx % DECODE_BLOCKS_PER_YIELD == 0 {
                yield_once().await;
            }
            self.decode_block(
                layout,
                block,
                &bytes[block.bytes.clone()],
                None,
                &mut postings,
            )?;
        }
        if let Some(last) = postings.last() {
            if let Some(start) = last.position_offset {
                self.layout.as_mut().unwrap().total_positions =
                    Some(start + u64::from(last.term_freq));
            }
        }
        Ok(postings)
    }

    /// Return the matching postings for a sorted batch of unique document IDs.
    pub(crate) async fn read_docs(
        &mut self,
        mut candidates: &[DocId],
    ) -> io::Result<Vec<CandidatePosting>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        if candidates.last() >= Some(&TERMINATED)
            || candidates.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "posting candidates must be sorted unique document IDs",
            ));
        }
        if self.layout.is_none() {
            let plan = self.plan_docs(candidates.len());
            plan.record("postings");
            if plan.whole {
                self.load_whole().await?;
            } else {
                self.load().await?;
            }
        }
        let layout = self.layout.as_ref().unwrap();
        let mut selected = Vec::new();
        let mut blocks = layout.blocks.as_slice();
        while let Some(&candidate) = candidates.first() {
            let next_block = blocks.partition_point(|block| block.last_doc < candidate);
            let Some((block, remainder)) = blocks[next_block..].split_first() else {
                break;
            };
            let end = candidates.partition_point(|&doc| doc <= block.last_doc);
            selected.push((block, &candidates[..end]));
            candidates = &candidates[end..];
            blocks = remainder;
        }
        let ranges: Vec<_> = selected
            .iter()
            .map(|(block, _)| block.bytes.clone())
            .collect();
        let tail = layout.blocks.last().filter(|block| {
            layout.total_positions.is_none()
                && self.positions.is_some()
                && matches!(block.info, BlockInfo::VInt { .. })
                && ranges.last().is_some_and(|last| {
                    last.end <= block.bytes.start
                        && block.bytes.start - last.end <= self.options.max_merge_gap
                })
        });
        let (bytes, optional_tail) = read_blocks_with_optional_tail(
            &self.postings,
            &layout.prefix,
            &ranges,
            tail.map(|block| block.bytes.clone()),
        )
        .await?;
        let mut total = layout.total_positions;
        if let (Some(block), Some(bytes)) = (tail, optional_tail) {
            total = self.decode_block(layout, block, &bytes, Some(&[]), &mut Vec::new())?;
        }
        let mut postings = Vec::new();
        for (block_idx, ((block, candidates), bytes)) in selected.into_iter().zip(bytes).enumerate()
        {
            if block_idx != 0 && block_idx % DECODE_BLOCKS_PER_YIELD == 0 {
                yield_once().await;
            }
            let decoded_tail =
                self.decode_block(layout, block, &bytes, Some(candidates), &mut postings)?;
            if decoded_tail.is_some() {
                total = decoded_tail;
            }
        }
        self.layout.as_mut().unwrap().total_positions = total;
        Ok(postings)
    }

    /// Return the position count when the skip table or an already-read tail determines it.
    pub(crate) fn total_positions(&self) -> Option<u64> {
        self.layout
            .as_ref()
            .and_then(|layout| layout.total_positions)
    }

    pub(crate) fn position_read_bytes(&self, hits: &[CandidatePosting]) -> io::Result<usize> {
        self.positions.as_ref().map_or(Ok(0), |reader| {
            reader.read_upper_bound(&position_ranges(hits), self.total_positions())
        })
    }

    /// Fetch the selected compressed position blocks for later decoding in bounded batches.
    pub(crate) async fn prefetch_positions(&mut self, hits: &[CandidatePosting]) -> io::Result<()> {
        let total_positions = self.total_positions();
        let Some(reader) = self.positions.as_mut() else {
            return Ok(());
        };
        reader
            .prefetch(&position_ranges(hits), total_positions)
            .await
    }

    /// Read positions only for the retained matches from this reader, in document order.
    pub(crate) async fn read_positions(&mut self, hits: &mut [CandidatePosting]) -> io::Result<()> {
        let total_positions = self.total_positions();
        let Some(reader) = self.positions.as_mut() else {
            return Ok(());
        };
        let ranges = position_ranges(hits);
        let positions = reader.read_positions(&ranges, total_positions).await?;
        for (hit, positions) in hits
            .iter_mut()
            .filter(|hit| hit.position_offset.is_some())
            .zip(positions)
        {
            hit.positions = positions;
        }
        Ok(())
    }

    fn decode_block(
        &self,
        layout: &PostingsLayout,
        block: &PostingBlock,
        mut bytes: &[u8],
        candidates: Option<&[DocId]>,
        output: &mut Vec<CandidatePosting>,
    ) -> io::Result<Option<u64>> {
        let mut doc_decoder = BlockDecoder::default();
        let mut freq_decoder = BlockDecoder::default();
        let mut tail_docs = [0; COMPRESSION_BLOCK_SIZE];
        let mut tail_freqs = [0; COMPRESSION_BLOCK_SIZE];
        let mut record_option = layout.record_option;
        let read_freqs = self.requested_option.has_freq();
        let mut add_one = false;
        let (docs, freqs, expected_tf_sum): (&[u32], &[u32], Option<u32>) = match block.info {
            BlockInfo::BitPacked {
                doc_num_bits,
                tf_num_bits,
                strict_delta_encoded,
                tf_sum,
                ..
            } => {
                let consumed = doc_decoder.uncompress_block_sorted(
                    bytes,
                    block.previous_doc,
                    doc_num_bits,
                    strict_delta_encoded,
                );
                let mut freqs = &[][..];
                if read_freqs && record_option.has_freq() {
                    // Check the offset on the decoded value below to reject overflow.
                    freq_decoder.uncompress_block_unsorted(&bytes[consumed..], tf_num_bits, false);
                    freqs = freq_decoder.output_array();
                    add_one = strict_delta_encoded;
                }
                let docs = doc_decoder.output_array();
                if docs.last() != Some(&block.last_doc) {
                    return Err(invalid_data("packed postings disagree with skip table"));
                }
                (docs, freqs, record_option.has_positions().then_some(tf_sum))
            }
            BlockInfo::VInt { num_docs } => {
                let mut doc = block.previous_doc;
                let num_docs = num_docs as usize;
                let docs = &mut tail_docs[..num_docs];
                for value in docs.iter_mut() {
                    doc = doc
                        .checked_add(read_vint(&mut bytes, 32)? as u32)
                        .ok_or_else(|| invalid_data("postings doc delta overflow"))?;
                    *value = doc;
                }
                let mut freqs = &[][..];
                if record_option.has_freq() {
                    if bytes.is_empty() && layout.blocks.len() == 1 {
                        // Short numerical JSON terms have no skip table to indicate that the
                        // term was encoded without frequencies.
                        record_option = IndexRecordOption::Basic;
                    } else {
                        let tail_freqs = &mut tail_freqs[..num_docs];
                        for freq in tail_freqs.iter_mut() {
                            *freq = read_vint(&mut bytes, 32)? as u32;
                            if *freq == 0 {
                                return Err(invalid_data("invalid postings term frequency"));
                            }
                        }
                        if read_freqs {
                            freqs = tail_freqs;
                        }
                    }
                }
                if !bytes.is_empty() {
                    return Err(invalid_data("unexpected bytes after postings tail"));
                }
                (&*docs, freqs, None)
            }
        };
        if docs.last() >= Some(&TERMINATED)
            || docs.windows(2).any(|pair| pair[0] >= pair[1])
            || (block.previous_doc != 0 && docs[0] <= block.previous_doc)
        {
            return Err(invalid_data("invalid document order in postings block"));
        }

        if self.positions.is_some() && !record_option.has_positions() {
            return Err(invalid_data(
                "frequencies and positions missing for positional postings",
            ));
        }
        let mut candidates = candidates;
        let mut position_offset = block.position_offset;
        for (i, &doc) in docs.iter().enumerate() {
            let tf = freqs
                .get(i)
                .map(|&tf| {
                    tf.checked_add(u32::from(add_one))
                        .filter(|&tf| tf != 0)
                        .ok_or_else(|| invalid_data("invalid postings term frequency"))
                })
                .transpose()?
                .unwrap_or(1);
            let matched = candidates.as_mut().is_none_or(|remaining| {
                *remaining = &remaining[remaining.partition_point(|&target| target < doc)..];
                remaining.first() == Some(&doc)
            });
            if matched {
                output.push(CandidatePosting {
                    doc,
                    term_freq: tf,
                    positions: Vec::new(),
                    position_offset: (record_option.has_positions() && self.positions.is_some())
                        .then_some(position_offset),
                });
            }
            position_offset += u64::from(tf);
        }
        if let Some(expected_tf_sum) = expected_tf_sum {
            if read_freqs && position_offset - block.position_offset != u64::from(expected_tf_sum) {
                return Err(invalid_data(
                    "postings frequency sum disagrees with skip table",
                ));
            }
        }
        Ok((record_option.has_positions()
            && read_freqs
            && matches!(block.info, BlockInfo::VInt { .. }))
        .then_some(position_offset))
    }
}

fn position_ranges(hits: &[CandidatePosting]) -> Vec<Range<u64>> {
    hits.iter()
        .filter_map(|hit| {
            let start = hit.position_offset?;
            Some(start..start + u64::from(hit.term_freq))
        })
        .collect()
}
