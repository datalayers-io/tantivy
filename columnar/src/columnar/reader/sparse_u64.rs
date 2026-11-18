use std::{io, ops::Range};

use common::{HasLen, OwnedBytes};
use sstable::{RangeSSTable, SSTable};
use tantivy_bitpacker::compute_num_bits;

use super::ColumnarReader;
use crate::column::open_column_u64;
use crate::column_values::CodecType;
use crate::{Cardinality, ColumnType};

// One cardinality byte, one codec byte, three u64 VInts and one u32 VInt.
const MAX_HEADER: usize = 2 + 3 * 10 + 5;
const INDEX_LENGTH_BYTES: usize = size_of::<u32>();

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

struct PackedColumn {
    min: u64,
    gcd: u64,
    max_delta: u64,
    width: usize,
    payload: usize,
}

impl PackedColumn {
    fn parse(
        head: &[u8],
        index_len: usize,
        rows: u32,
        column_len: usize,
    ) -> io::Result<Option<Self>> {
        if head.len() < 2 || index_len >= column_len - INDEX_LENGTH_BYTES {
            return Err(invalid("invalid column header length"));
        }
        if head[0] != Cardinality::Full as u8 || head[1] != CodecType::Bitpacked as u8 {
            return Ok(None);
        }
        if index_len != 1 {
            return Err(invalid("invalid full column index length"));
        }
        let mut rest = &head[2..];
        let mut read = || {
            let mut value = 0u64;
            for shift in (0..64).step_by(7) {
                let (&byte, tail) = rest
                    .split_first()
                    .ok_or_else(|| invalid("truncated column statistic"))?;
                rest = tail;
                if shift == 63 && byte & 0x7f > 1 {
                    return Err(invalid("column statistic overflow"));
                }
                value |= u64::from(byte & 0x7f) << shift;
                if byte >= 0x80 {
                    return Ok(value);
                }
            }
            Err(invalid("unterminated column statistic"))
        };
        let min = read()?;
        let gcd = read()?;
        let max_delta = read()?;
        if read()? != u64::from(rows)
            || gcd == 0
            || max_delta
                .checked_mul(gcd)
                .and_then(|v| min.checked_add(v))
                .is_none()
        {
            return Err(invalid("invalid full column statistics"));
        }
        let width = usize::from(compute_num_bits(max_delta));
        let payload = head.len() - rest.len();
        let payload_bytes = (u64::from(rows) * width as u64).div_ceil(8);
        if payload_bytes != (column_len - INDEX_LENGTH_BYTES).saturating_sub(payload) as u64 {
            return Err(invalid("bitpacked column payload length mismatch"));
        }
        Ok(Some(Self {
            min,
            gcd,
            max_delta,
            width,
            payload,
        }))
    }

    fn byte_range(&self, doc: u32) -> Range<usize> {
        let start = u64::from(doc) * self.width as u64;
        self.payload + (start / 8) as usize
            ..self.payload + (start + self.width as u64).div_ceil(8) as usize
    }

    fn value(&self, doc: u32, bytes: &[u8]) -> io::Result<u64> {
        let mut word = [0u8; 8];
        if bytes.len() > word.len() {
            return Err(invalid("invalid bitpacked value width"));
        }
        word[..bytes.len()].copy_from_slice(bytes);
        let shift = (u64::from(doc) * self.width as u64) % 8;
        let mask = u64::MAX.checked_shr((64 - self.width) as u32).unwrap_or(0);
        let delta = (u64::from_le_bytes(word) >> shift) & mask;
        if delta > self.max_delta {
            return Err(invalid("bitpacked value exceeds column statistics"));
        }
        Ok(self.min + delta * self.gcd)
    }
}

impl ColumnarReader {
    /// Fetches u64 values in the caller's document order. For a single column
    /// the column start and dictionary start are known from the opened footer.
    /// This permits a single metadata wave before gathering packed values.
    pub async fn first_u64_values(
        &self,
        name: &str,
        docs: &[u32],
        default_value: u64,
        max_merge_gap: usize,
        min_saving_per_extra_get: usize,
    ) -> io::Result<Vec<u64>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        if docs.iter().any(|&doc| doc >= self.num_docs) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "document outside column",
            ));
        }
        if self.num_columns() != 1 {
            let handle = self
                .read_columns_async(name)
                .await?
                .into_iter()
                .find(|col| col.column_type == ColumnType::U64)
                .ok_or_else(|| invalid("missing u64 fast field"))?;
            let column = open_column_u64::<u64>(
                handle.file_slice.read_bytes_async().await?,
                self.format_version,
            )?;
            return Ok(docs
                .iter()
                .map(|&doc| column.first(doc).unwrap_or(default_value))
                .collect());
        }
        let end = self.column_data.len();
        let index_end = end
            .checked_add(self.column_dictionary.sstable_slice.len())
            .filter(|&n| n <= self.file.len())
            .ok_or_else(|| invalid("invalid dictionary length"))?;
        let body_end = end
            .checked_sub(INDEX_LENGTH_BYTES)
            .ok_or_else(|| invalid("truncated column"))?;
        let mut head_end = MAX_HEADER.min(body_end);
        // A full column stores one fixed-width value per row. Its length and
        // row count therefore bound the width even before the stats arrive.
        let width_bound = ((body_end as u128 * 8) / u128::from(self.num_docs)).min(64) as u64;
        for &doc in docs {
            let value_end = ((u64::from(doc) + 1) * width_bound).div_ceil(8);
            let bound = MAX_HEADER.saturating_add(usize::try_from(value_end).unwrap_or(usize::MAX));
            if bound <= MAX_HEADER.saturating_add(max_merge_gap) {
                head_end = head_end.max(bound.min(body_end));
            }
        }
        // Each distinct point might require one GET, plus the two metadata
        // requests. Bound gap bytes too before paying for either probe.
        let sparse_bytes = head_end
            .saturating_add(index_end - body_end)
            .saturating_add(
                docs.len()
                    .saturating_mul(8usize.saturating_add(max_merge_gap)),
            );
        let extra_gets = docs.len().saturating_add(1);
        let use_whole = sparse_bytes >= index_end
            || index_end - sparse_bytes < extra_gets.saturating_mul(min_saving_per_extra_get)
            || head_end >= body_end;
        let (head, tail, whole) = if use_whole {
            let bytes = self.file.read_bytes_slice_async(0..index_end).await?;
            if bytes.len() != index_end {
                return Err(invalid("short whole column read"));
            }
            (
                bytes.slice(0..head_end),
                bytes.slice(body_end..index_end),
                Some(bytes.slice(0..end)),
            )
        } else {
            let ranges = vec![0..head_end, body_end..index_end];
            let [head, tail]: [_; 2] = self
                .file
                .read_bytes_batch_async(ranges)
                .await?
                .try_into()
                .map_err(|_| invalid("missing metadata block"))?;
            if head.len() != head_end || tail.len() != index_end - body_end {
                return Err(invalid("short column metadata read"));
            }
            (head, tail, None)
        };
        let mut dict = RangeSSTable::reader(tail.slice(INDEX_LENGTH_BYTES..tail.len()));
        let mut key = name.as_bytes().to_vec();
        key.extend([0, ColumnType::U64.to_code()]);
        if !dict.advance()?
            || dict.key() != key
            || *dict.value() != (0..end as u64)
            || dict.advance()?
        {
            return Err(invalid("invalid single u64 column dictionary entry"));
        }
        let index_len = u32::from_le_bytes(tail[..4].try_into().unwrap()) as usize;
        let layout = PackedColumn::parse(&head, index_len, self.num_docs, end)?;
        if let (Some(layout), None) = (&layout, &whole) {
            let ranges: Vec<_> = docs
                .iter()
                .map(|&doc| layout.byte_range(doc))
                .filter(|r| !r.is_empty() && r.end > head_end)
                .map(|r| r.start.max(head_end)..r.end)
                .collect();
            let loaded = if ranges.is_empty() {
                Vec::new()
            } else {
                self.column_data.read_bytes_batch_async(ranges).await?
            };
            let mut loaded = loaded.into_iter();
            let mut values = Vec::with_capacity(docs.len());
            for &doc in docs {
                let range = layout.byte_range(doc);
                let mut data = [0; 8];
                let prefix = range.end.min(head_end).saturating_sub(range.start);
                if prefix > 0 {
                    data[..prefix].copy_from_slice(&head[range.start..range.start + prefix]);
                }
                if range.len() > prefix {
                    let rest = loaded
                        .next()
                        .ok_or_else(|| invalid("missing packed value"))?;
                    if rest.len() != range.len() - prefix {
                        return Err(invalid("short packed value"));
                    }
                    data[prefix..range.len()].copy_from_slice(&rest);
                }
                values.push(layout.value(doc, &data[..range.len()])?);
            }
            if loaded.next().is_some() {
                return Err(invalid("unexpected packed values in batch"));
            }
            return Ok(values);
        }
        let bytes = if let Some(whole) = whole {
            whole
        } else {
            let middle = self
                .column_data
                .read_bytes_slice_async(head_end..body_end)
                .await?;
            if middle.len() != body_end - head_end {
                return Err(invalid("short column fallback read"));
            }
            let mut bytes = Vec::with_capacity(end);
            bytes.extend_from_slice(&head);
            bytes.extend_from_slice(&middle);
            bytes.extend_from_slice(&tail[..4]);
            OwnedBytes::new(bytes)
        };
        let column = open_column_u64::<u64>(bytes, self.format_version)?;
        Ok(docs
            .iter()
            .map(|&doc| column.first(doc).unwrap_or(default_value))
            .collect())
    }
}
