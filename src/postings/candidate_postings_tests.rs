use std::future::Future;
use std::io;
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

use futures::executor::block_on;

use super::candidate_postings::CandidatePostingsReader;
use super::serializer::PostingsSerializer;
use super::{BlockSegmentPostings, Postings, SegmentPostings};
use crate::directory::{FileHandle, FileSlice, OwnedBytes};
use crate::positions::{PositionReader, PositionSerializer};
use crate::schema::IndexRecordOption::{self, Basic, WithFreqs, WithFreqsAndPositions};
use crate::{DocId, DocSet, HasLen, TERMINATED};

fn encoded_term(doc_freq: u32, option: IndexRecordOption, has_freqs: bool) -> (Vec<u8>, Vec<u8>) {
    let mut postings = PostingsSerializer::new(0.0, option, None);
    let mut position_bytes = Vec::new();
    let mut positions = PositionSerializer::new(&mut position_bytes);
    postings.new_term(doc_freq, has_freqs);
    for i in 0..doc_freq {
        // The second block has a nonzero delta base. Some docs' positions cross a block boundary.
        let tf = if i == 130 { 190 } else { 1 + i % 11 };
        postings.write_doc(i * 3, tf);
        if option.has_positions() && has_freqs {
            let mut deltas = vec![1; tf as usize];
            deltas[0] = i % 7;
            positions.write_positions_delta(&deltas);
        }
    }
    let mut posting_bytes = Vec::new();
    postings.close_term(doc_freq, &mut posting_bytes).unwrap();
    positions.close_term().unwrap();
    positions.close().unwrap();
    (posting_bytes, position_bytes)
}

#[test]
fn sparse_candidate_docs_and_positions_match_original_postings() {
    for doc_freq in [0, 1, 127, 128, 129, 256, 257, 384, 385] {
        for (recorded, requested, has_freqs) in [
            (Basic, WithFreqsAndPositions, false),
            (WithFreqs, WithFreqs, true),
            (WithFreqsAndPositions, Basic, true),
            (WithFreqsAndPositions, WithFreqs, false),
            (WithFreqsAndPositions, WithFreqsAndPositions, true),
        ] {
            let (postings, positions) = encoded_term(doc_freq, recorded, has_freqs);
            let postings = FileSlice::from(postings);
            let with_positions = recorded.has_positions() && requested.has_positions();
            let mut original = SegmentPostings::from_block_postings(
                BlockSegmentPostings::open(doc_freq, postings.clone(), recorded, requested)
                    .unwrap(),
                with_positions
                    .then(|| PositionReader::open(OwnedBytes::new(positions.clone())).unwrap()),
            );
            let mut reader = CandidatePostingsReader::new(
                doc_freq,
                postings,
                with_positions.then(|| FileSlice::from(positions)),
                recorded,
                requested,
            );
            let batch_a = [0, 1, 8, 64, 123, 378, 381, 382, 384, 387, 390, 391];
            let batch_b = [
                719,
                762,
                765,
                766,
                768,
                771,
                1149,
                1150,
                1152,
                1155,
                TERMINATED - 1,
            ];
            let mut all = block_on(reader.all_docs()).unwrap();
            block_on(reader.read_positions(&mut all)).unwrap();
            let batch_hits: Vec<_> = [batch_a.as_slice(), batch_b.as_slice()]
                .into_iter()
                .flat_map(|batch| {
                    let mut matches = block_on(reader.read_docs(batch)).unwrap();
                    block_on(reader.read_positions(&mut matches)).unwrap();
                    matches
                })
                .collect();
            let mut expected = Vec::new();
            let mut expected_all = Vec::new();
            while original.doc() != TERMINATED {
                let mut positions = Vec::new();
                original.positions(&mut positions);
                let hit = (original.doc(), original.term_freq(), positions);
                if batch_a.contains(&hit.0) || batch_b.contains(&hit.0) {
                    expected.push(hit.clone());
                }
                expected_all.push(hit);
                original.advance();
            }
            assert_eq!(
                batch_hits
                    .into_iter()
                    .map(|p| (p.doc, p.term_freq, p.positions))
                    .collect::<Vec<_>>(),
                expected,
                "doc_freq={doc_freq}, {recorded:?}, {requested:?}, has_freqs={has_freqs}"
            );
            assert_eq!(
                all.into_iter()
                    .map(|p| (p.doc, p.term_freq, p.positions))
                    .collect::<Vec<_>>(),
                expected_all
            );
        }
    }
}

#[derive(Debug)]
struct AsyncFile {
    bytes: OwnedBytes,
    ranges: Mutex<Vec<Range<usize>>>,
    short_read_from: Option<usize>,
}

impl HasLen for AsyncFile {
    fn len(&self) -> usize {
        self.bytes.len()
    }
}

impl FileHandle for AsyncFile {
    fn read_bytes(&self, _: Range<usize>) -> io::Result<OwnedBytes> {
        panic!("candidate reader issued a synchronous file read");
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
            self.ranges.lock().unwrap().extend(ranges.iter().cloned());
            let result = ranges
                .into_iter()
                .map(|mut range| {
                    if self
                        .short_read_from
                        .is_some_and(|start| range.start >= start)
                        && !range.is_empty()
                    {
                        range.end -= 1;
                    }
                    self.bytes.slice(range)
                })
                .collect();
            Ok(result)
        })
    }
}

fn async_file(bytes: Vec<u8>, short_read_from: Option<usize>) -> (Arc<AsyncFile>, FileSlice) {
    let file = Arc::new(AsyncFile {
        bytes: OwnedBytes::new(bytes),
        ranges: Mutex::default(),
        short_read_from,
    });
    (file.clone(), FileSlice::new(file))
}

#[test]
fn sparse_reads_skip_untouched_payloads_and_reuse_metadata() {
    let (postings, positions) = encoded_term(16_384, WithFreqsAndPositions, true);
    let (postings_file, postings) = async_file(postings, None);
    let (positions_file, positions) = async_file(positions, None);
    let mut reader = CandidatePostingsReader::new(
        16_384,
        postings,
        Some(positions),
        WithFreqsAndPositions,
        WithFreqsAndPositions,
    );
    let mut first = block_on(reader.read_docs(&[0, 1, 2, 3, 4, 5])).unwrap();
    assert_eq!(first.iter().map(|p| p.doc).collect::<Vec<DocId>>(), [0, 3]);
    assert!(positions_file.ranges.lock().unwrap().is_empty());
    block_on(reader.read_positions(&mut first)).unwrap();
    assert_eq!(first[0].positions, [0]);
    assert_eq!(first[1].positions, [1, 2]);
    let first_reads = postings_file.ranges.lock().unwrap().len();
    let first_position_reads = positions_file.ranges.lock().unwrap().len();
    let mut second = block_on(reader.read_docs(&[49_149, 49_150])).unwrap();
    block_on(reader.read_positions(&mut second)).unwrap();
    assert_eq!(second[0].doc, 49_149);
    for (file, original_read_count) in [
        (postings_file, first_reads),
        (positions_file, first_position_reads),
    ] {
        let reads = file.ranges.lock().unwrap();
        assert_eq!(reads.iter().filter(|range| range.start == 0).count(), 1);
        assert!(reads.len() > original_read_count);
        assert!(reads.iter().map(|range| range.len()).sum::<usize>() < file.len() / 3);
    }
}

#[test]
fn malformed_metadata_and_payload_are_errors() {
    for (doc_freq, bytes) in [
        (128, vec![0; 12]),                      // Unterminated skip-length vint.
        (128, vec![0x85, 0, 0, 0, 0, 0x42]),     // Declares a nonempty block with no payload.
        (1, vec![0]),                            // Unterminated doc delta.
        (1, vec![0xff, 0xff, 0xff, 0xff, 0xff]), // Overflowed doc delta.
    ] {
        let mut reader =
            CandidatePostingsReader::new(doc_freq, FileSlice::from(bytes), None, Basic, Basic);
        let error = block_on(reader.read_docs(&[0])).expect_err("must reject corruption");
        assert!(matches!(
            error.kind(),
            io::ErrorKind::InvalidData | io::ErrorKind::UnexpectedEof
        ));
    }

    let (postings, _) = encoded_term(1, WithFreqsAndPositions, true);
    let mut reader = CandidatePostingsReader::new(
        1,
        FileSlice::from(postings),
        Some(FileSlice::from(vec![0x80, 0x00])),
        WithFreqsAndPositions,
        WithFreqsAndPositions,
    );
    let mut hits = block_on(reader.read_docs(&[0])).unwrap();
    assert!(block_on(reader.read_positions(&mut hits)).is_err());
}

#[test]
fn short_async_posting_and_position_reads_are_errors() {
    let (postings, positions) = encoded_term(1, WithFreqsAndPositions, true);
    let (_, short_postings) = async_file(postings.clone(), Some(0));
    let mut reader =
        CandidatePostingsReader::new(1, short_postings, None, WithFreqsAndPositions, WithFreqs);
    let error = block_on(reader.read_docs(&[0])).expect_err("must reject short read");
    assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);

    let (_, short_positions) = async_file(positions, Some(0));
    let mut reader = CandidatePostingsReader::new(
        1,
        FileSlice::from(postings),
        Some(short_positions),
        WithFreqsAndPositions,
        WithFreqsAndPositions,
    );
    let mut hits = block_on(reader.read_docs(&[0])).unwrap();
    let error = block_on(reader.read_positions(&mut hits)).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
}

#[test]
fn selected_postings_errors_are_preserved_after_successful_metadata_read() {
    let doc_freq = 8192;
    let (postings, _) = encoded_term(doc_freq, WithFreqs, true);
    let (file, postings) = async_file(postings, Some(1));
    let mut reader = CandidatePostingsReader::new(doc_freq, postings, None, WithFreqs, WithFreqs);
    block_on(reader.load()).unwrap();
    assert_eq!(file.ranges.lock().unwrap().len(), 1);
    let error = block_on(reader.read_docs(&[0])).expect_err("must reject a short payload");
    assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    assert_eq!(file.ranges.lock().unwrap().len(), 2);
}

#[test]
fn terms_with_zero_width_blocks_need_no_payload_reads() {
    let mut posting_bytes = Vec::new();
    let mut positions_bytes = Vec::new();
    let mut postings = PostingsSerializer::new(0.0, WithFreqsAndPositions, None);
    let mut positions = PositionSerializer::new(&mut positions_bytes);
    postings.new_term(256, true);
    for doc in 0..256 {
        postings.write_doc(doc, 1);
        positions.write_positions_delta(&[0]);
    }
    postings.close_term(256, &mut posting_bytes).unwrap();
    positions.close_term().unwrap();
    positions.close().unwrap();
    let (posting_file, posting_slice) = async_file(posting_bytes, None);
    let (position_file, position_slice) = async_file(positions_bytes, None);
    let mut reader = CandidatePostingsReader::new(
        256,
        posting_slice,
        Some(position_slice),
        WithFreqsAndPositions,
        WithFreqsAndPositions,
    );
    let mut hits = block_on(reader.read_docs(&[0, 127, 128, 255, 256])).unwrap();
    block_on(reader.read_positions(&mut hits)).unwrap();
    assert_eq!(
        hits.iter()
            .map(|hit| (hit.doc, hit.term_freq, hit.positions.as_slice()))
            .collect::<Vec<_>>(),
        [
            (0, 1, [0].as_slice()),
            (127, 1, &[0]),
            (128, 1, &[0]),
            (255, 1, &[0])
        ],
    );
    assert_eq!(posting_file.ranges.lock().unwrap().len(), 1);
    assert_eq!(position_file.ranges.lock().unwrap().len(), 1);
}

#[test]
fn complete_postings_blocks_give_the_position_header_length() {
    let doc_freq = 91_136;
    let (postings, positions) = encoded_term(doc_freq, WithFreqsAndPositions, true);
    let (position_file, position_slice) = async_file(positions, None);
    let mut reader = CandidatePostingsReader::new(
        doc_freq,
        FileSlice::from(postings),
        Some(position_slice),
        WithFreqsAndPositions,
        WithFreqsAndPositions,
    );
    let mut hits = block_on(reader.read_docs(&[0])).unwrap();
    block_on(reader.read_positions(&mut hits)).unwrap();
    assert_eq!(hits[0].positions, [0]);
    let reads = position_file.ranges.lock().unwrap();
    assert_eq!(
        reads.len(),
        2,
        "read the large metadata and one payload block"
    );
    assert!(reads[0].len() > 4 * 1024);
    assert_eq!(reads[0].end, reads[1].start);
}

#[test]
fn small_packed_terms_share_one_read_for_metadata_and_payload() {
    let (postings, positions) = encoded_term(384, WithFreqsAndPositions, true);
    let (postings_file, postings) = async_file(postings, None);
    let (positions_file, positions) = async_file(positions, None);
    let mut reader = CandidatePostingsReader::new(
        384,
        postings,
        Some(positions),
        WithFreqsAndPositions,
        WithFreqsAndPositions,
    );
    let mut hits = block_on(reader.read_docs(&[0, 1149])).unwrap();
    block_on(reader.read_positions(&mut hits)).unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.doc).collect::<Vec<_>>(),
        [0, 1149]
    );
    assert_eq!(hits[0].positions, [0]);
    assert_eq!(postings_file.ranges.lock().unwrap().len(), 1);
    assert_eq!(positions_file.ranges.lock().unwrap().len(), 1);
}

#[test]
fn prefetched_position_blocks_are_reused_across_candidate_batches() {
    let doc_freq = 16_384;
    let (postings, positions) = encoded_term(doc_freq, WithFreqsAndPositions, true);
    let (positions_file, positions) = async_file(positions, None);
    let mut reader = CandidatePostingsReader::new(
        doc_freq,
        FileSlice::from(postings),
        Some(positions),
        WithFreqsAndPositions,
        WithFreqsAndPositions,
    );
    let mut hits = block_on(reader.read_docs(&[0, 24_576, 49_149])).unwrap();
    block_on(reader.prefetch_positions(&hits)).unwrap();
    let reads = positions_file.ranges.lock().unwrap().clone();
    assert!(reads.len() > 1);
    for hit in hits.chunks_mut(1) {
        block_on(reader.read_positions(hit)).unwrap();
    }
    assert_eq!(*positions_file.ranges.lock().unwrap(), reads);
    assert_eq!(hits[0].positions, [0]);
    assert_eq!(hits[1].positions[0], 8192 % 7);
    assert_eq!(hits[2].positions[0], 16_383 % 7);
}

#[test]
fn driving_postings_and_positions_use_batch_only_file_handle() {
    let (postings, positions) = encoded_term(129, WithFreqsAndPositions, true);
    let (postings_file, postings) = async_file(postings, None);
    let (_, positions) = async_file(positions, None);
    let mut reader = CandidatePostingsReader::new(
        129,
        postings,
        Some(positions),
        WithFreqsAndPositions,
        WithFreqsAndPositions,
    );
    let mut hits = block_on(reader.all_docs()).unwrap();
    block_on(reader.read_positions(&mut hits)).unwrap();
    assert_eq!(hits.len(), 129);
    assert_eq!(reader.total_positions(), Some(762));
    assert_eq!(hits[128].doc, 384);
    assert_eq!(hits[128].positions[0], 128 % 7);
    assert_eq!(
        *postings_file.ranges.lock().unwrap(),
        std::iter::once(0..postings_file.len()).collect::<Vec<_>>()
    );
}

#[test]
fn posting_decoding_yields_with_immediately_ready_batch_io() {
    #[derive(Default)]
    struct Notification(AtomicBool);

    impl Wake for Notification {
        fn wake(self: Arc<Self>) {
            self.0.store(true, Ordering::Relaxed);
        }
    }

    fn poll_across_yield<F: Future>(future: F) -> F::Output {
        let mut future = std::pin::pin!(future);
        let notification = Arc::new(Notification::default());
        let waker = Waker::from(notification.clone());
        let mut context = Context::from_waker(&waker);
        assert!(future.as_mut().poll(&mut context).is_pending());
        assert!(notification.0.load(Ordering::Relaxed));
        let Poll::Ready(value) = future.as_mut().poll(&mut context) else {
            panic!("posting decode should finish after one cooperative yield");
        };
        value
    }

    let doc_freq = 129 * 128;
    let (postings, _) = encoded_term(doc_freq, WithFreqs, true);
    let (_, postings) = async_file(postings, None);
    let mut reader = CandidatePostingsReader::new(doc_freq, postings, None, WithFreqs, WithFreqs);
    let all = poll_across_yield(reader.all_docs()).unwrap();
    assert_eq!(all.len(), doc_freq as usize);

    let candidates: Vec<_> = (0..doc_freq).step_by(128).map(|doc| doc * 3).collect();
    let selected = poll_across_yield(reader.read_docs(&candidates)).unwrap();
    assert_eq!(selected.len(), 129);
    assert_eq!(selected[128].doc, candidates[128]);
}
