use std::future::Future;
use std::io;
use std::ops::Range;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use columnar::{ColumnType, ColumnarReader, ColumnarWriter};

use crate::HasLen;
use crate::directory::{FileHandle, FileSlice, OwnedBytes};

#[derive(Debug)]
struct CountingFile {
    data: OwnedBytes,
    waves: Mutex<Vec<Vec<Range<usize>>>>,
}

impl HasLen for CountingFile {
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl FileHandle for CountingFile {
    fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
        Ok(self.data.slice(range))
    }
    fn read_bytes_async<'a, 'b>(
        &'a self,
        range: Range<usize>,
    ) -> Pin<Box<dyn Future<Output = io::Result<OwnedBytes>> + Send + 'b>>
    where
        'a: 'b,
        Self: 'b,
    {
        Box::pin(async move {
            self.waves.lock().unwrap().push(vec![range.clone()]);
            self.read_bytes(range)
        })
    }
    fn read_bytes_batch_async<'a, 'b>(
        &'a self,
        ranges: Vec<Range<usize>>,
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<OwnedBytes>>> + Send + 'b>>
    where
        'a: 'b,
        Self: 'b,
    {
        Box::pin(async move {
            self.waves.lock().unwrap().push(ranges.clone());
            ranges.into_iter().map(|r| self.read_bytes(r)).collect()
        })
    }
}

fn column(values: &[Option<u64>]) -> (Arc<CountingFile>, ColumnarReader) {
    let mut writer = ColumnarWriter::default();
    writer.record_column_type("key", ColumnType::U64, false);
    for (doc, val) in values.iter().enumerate() {
        if let Some(value) = val {
            writer.record_numerical(doc as u32, "key", *value);
        }
    }
    let mut data = Vec::new();
    writer
        .serialize(values.len() as u32, None, &mut data)
        .unwrap();
    let counted = Arc::new(CountingFile {
        data: OwnedBytes::new(data),
        waves: Mutex::default(),
    });
    let column = ColumnarReader::open(FileSlice::new(counted.clone())).unwrap();
    (counted, column)
}

#[test]
fn sparse_u64_waves_and_order() {
    let values: Vec<_> = (0..10007u32)
        .map(|i| Some(71 + u64::from((i * 2513) % 10007) * 3))
        .collect();
    let (file, reader) = column(&values);
    let docs = [9823, 1, 3603, 9823, 10006];
    let result =
        futures::executor::block_on(reader.first_u64_values("key", &docs, 0, 128, 256)).unwrap();
    assert_eq!(result, docs.map(|i| values[i as usize].unwrap()));
    let waves = file.waves.lock().unwrap();
    assert_eq!(waves.len(), 2, "{waves:?}");
    assert_eq!(waves[0].len(), 2);
    assert_eq!(waves[0][0].start, 0);
    let tail = &waves[0][1];
    assert!(tail.start > waves[0][0].end);
    assert_eq!(&file.data[tail.start..tail.start + 4], &1u32.to_le_bytes());
    assert!(tail.len() > 4);
    assert!(
        waves[1]
            .iter()
            .all(|r| r.start >= waves[0][0].end && r.end <= tail.start && r.len() <= 8)
    );
}

#[test]
fn sparse_u64_constant_wide_and_optional() {
    let docs = [2047, 0, 511, 0];
    for values in [
        vec![Some(314159); 2048],
        (0..2048u64)
            .map(|i| Some(i.wrapping_mul(0xc983b9a8f87212d3)))
            .collect(),
        (0..2048u64)
            .map(|i| (i % 2 == 1).then_some(i * 3 + 2))
            .collect(),
    ] {
        let (_, reader) = column(&values);
        let expected = docs.map(|i| values[i as usize].unwrap_or(37));
        assert_eq!(
            futures::executor::block_on(reader.first_u64_values("key", &docs, 37, 0, 0)).unwrap(),
            expected
        );
    }
}

#[test]
fn sparse_u64_short_column_uses_one_whole_request_and_validates_docs() {
    let (file, reader) = column(&[Some(450), Some(33), Some(123)]);
    let read = |docs: &[u32]| {
        futures::executor::block_on(reader.first_u64_values("key", docs, 0, 4096, 65536))
    };
    assert!(read(&[]).unwrap().is_empty());
    assert!(read(&[3]).is_err());
    assert!(file.waves.lock().unwrap().is_empty());
    assert_eq!(read(&[2, 0]).unwrap(), [123, 450]);
    let waves = file.waves.lock().unwrap();
    assert_eq!(waves.len(), 1);
    assert_eq!(waves[0].len(), 1);
}
