//! Query-scoped asynchronous preparation of selective conjunctions.

use std::collections::HashMap;
use std::sync::Arc;

use futures_util::{stream, FutureExt, StreamExt, TryStreamExt};

use crate::fieldnorm::FieldNormReader;
use crate::index::SegmentId;
use crate::postings::block_read::yield_once;
pub use crate::postings::block_read::CandidateReadOptions;
use crate::postings::{CandidatePosting, Postings};
use crate::query::phrase_query::PhraseScorer;
use crate::query::{
    Bm25Weight, BooleanQuery, BoostQuery, EnableScoring, Explanation, Occur, PhraseQuery, Query,
    Scorer, TermQuery, Weight,
};
use crate::schema::{Field, IndexRecordOption};
use crate::{
    DocId, DocSet, Score, Searcher, SearcherGeneration, SegmentReader, TantivyError, Term,
    TERMINATED,
};

const CANDIDATE_BATCH_LEN: usize = 256;
const PREPARE_CONCURRENCY: usize = 8;
// A conservative preparation estimate, not a process RSS/cache limit.
const PREPARATION_BUDGET_BYTES: usize = 64 * 1024 * 1024;

enum CandidateShape<'a> {
    Phrase(&'a PhraseQuery),
    Intersection([&'a TermQuery; 2]),
}

fn shape(query: &dyn Query) -> Option<(CandidateShape<'_>, Score)> {
    let (query, boost) = if let Some(boosted) = query.downcast_ref::<BoostQuery>() {
        (boosted.query(), boosted.boost())
    } else {
        (query, 1.0)
    };
    if let Some(phrase) = query.downcast_ref::<PhraseQuery>() {
        return Some((CandidateShape::Phrase(phrase), boost));
    }
    let boolean = query.downcast_ref::<BooleanQuery>()?;
    if boolean.get_minimum_number_should_match() != 0 || boolean.clauses().len() != 2 {
        return None;
    }
    let terms: Option<Vec<_>> = boolean
        .clauses()
        .iter()
        .map(|(occur, query)| {
            if *occur != Occur::Must {
                return None;
            }
            let term = query.downcast_ref::<TermQuery>()?;
            (term.index_record_option() == IndexRecordOption::WithFreqs).then_some(term)
        })
        .collect();
    let terms = terms?;
    Some((CandidateShape::Intersection([terms[0], terms[1]]), boost))
}

/// Whether query preparation supports this exact query shape.
///
/// Supports phrases and two-term exact conjunctions, optionally under one boost.
/// Other queries must retain their regular warmup and execution path.
pub fn supports_candidate_query(query: &dyn Query) -> bool {
    shape(query).is_some()
}

#[derive(Clone, Debug)]
struct PreparedRow {
    doc: DocId,
    // Phrase frequency, or the two conjunction term frequencies.
    frequencies: [u32; 2],
    matches_unscored: bool,
}

/// Candidate matches prepared for one searcher snapshot.
///
/// This query owns its match evidence and still validates the original snapshot
/// when creating a weight. Call [`Self::is_empty`] before warming field norms or
/// collector fast fields when an empty result can be returned directly.
#[derive(Clone, Debug)]
pub struct PreparedCandidateQuery {
    generation: SearcherGeneration,
    terms: Vec<Term>,
    phrase: bool,
    boost: Score,
    rows: HashMap<SegmentId, Arc<[PreparedRow]>>,
}

impl PreparedCandidateQuery {
    /// Whether preparation proved there are no matches in either scoring mode.
    ///
    /// Checks actual prepared intersections and phrase matches, including the
    /// original frequency and existence semantics. A false result may include
    /// deleted documents, which the normal search pipeline subsequently filters.
    /// This evidence applies only to the snapshot used for preparation.
    pub fn is_empty(&self) -> bool {
        self.rows.values().all(|rows| rows.is_empty())
    }
}

/// Prepare supported queries using candidate-driven postings and positions reads.
///
/// The returned query must be searched with the same searcher snapshot. Its
/// synchronous scorer reads no postings or positions. Preparation does not load
/// field norms or fast fields; callers may skip their warmup for an empty result.
/// Otherwise warm those dependencies before searching. Position materialization
/// is limited to batches of 256 candidate documents. Retained document/frequency rows, intermediate term
/// intersections, and prefetched compressed position blocks are guarded by a
/// conservative 64 MiB preparation estimate. Exceeding it returns `None` for
/// legacy execution without truncating results. This is not a process RSS limit:
/// directory caches and transport coalescing overread are outside this estimate.
/// Returns `None` for unsupported shapes without reading index data.
pub async fn prepare_candidate_query(
    searcher: &Searcher,
    query: &dyn Query,
) -> crate::Result<Option<PreparedCandidateQuery>> {
    prepare_candidate_query_with_options(searcher, query, CandidateReadOptions::default()).await
}

/// Prepare candidates using the directory's actual batch transport limits.
///
/// The options affect only conservative cold-I/O admission. They do not change
/// matching, scores, or cache correctness. See [`prepare_candidate_query`].
pub async fn prepare_candidate_query_with_options(
    searcher: &Searcher,
    query: &dyn Query,
    options: CandidateReadOptions,
) -> crate::Result<Option<PreparedCandidateQuery>> {
    prepare_with_options_and_budget(searcher, query, options, PREPARATION_BUDGET_BYTES).await
}

#[cfg(test)]
async fn prepare_with_budget(
    searcher: &Searcher,
    query: &dyn Query,
    budget: usize,
) -> crate::Result<Option<PreparedCandidateQuery>> {
    prepare_with_options_and_budget(searcher, query, CandidateReadOptions::default(), budget).await
}

async fn prepare_with_options_and_budget(
    searcher: &Searcher,
    query: &dyn Query,
    options: CandidateReadOptions,
    budget: usize,
) -> crate::Result<Option<PreparedCandidateQuery>> {
    if options.max_read_bytes == 0 {
        return Err(TantivyError::InvalidArgument(
            "candidate max_read_bytes must be nonzero".into(),
        ));
    }
    let Some((shape, boost)) = shape(query) else {
        return Ok(None);
    };
    if let CandidateShape::Intersection(terms) = &shape {
        // A non-indexed fast field can use TermQuery's unscored range fallback,
        // while its scored weight rejects the query. Preserve both behaviors
        // instead of interpreting a missing inverted index as no matches.
        if terms.iter().any(|term| {
            !searcher
                .schema()
                .get_field_entry(term.term().field())
                .is_indexed()
        }) {
            return Ok(None);
        }
    }
    let (terms, phrase) = match &shape {
        CandidateShape::Phrase(phrase) => (phrase.phrase_terms(), true),
        CandidateShape::Intersection(terms) => (
            terms.iter().map(|term| term.term().clone()).collect(),
            false,
        ),
    };

    // Warm every term lookup before constructing any synchronous BM25 weight,
    // including segments with no intersection. Their statistics still matter.
    let mut segment_infos = Vec::new();
    for reader in searcher.segment_readers() {
        let mut futures = Vec::with_capacity(terms.len());
        for term in &terms {
            futures.push(
                async move {
                    let inverted = reader.inverted_index(term.field())?;
                    Ok::<_, TantivyError>(inverted.get_term_info_async(term).await?)
                }
                .boxed(),
            );
        }
        let infos: Vec<_> = stream::iter(futures)
            .buffered(PREPARE_CONCURRENCY)
            .try_collect()
            .await?;
        segment_infos.push(infos);
    }
    // Charge all segments before expanding the first leader. This includes
    // retained final rows, Vec capacity growth and intermediate term hits.
    let mut estimated_bytes = 0usize;
    for infos in &segment_infos {
        if infos.iter().any(Option::is_none) {
            continue;
        }
        let infos: Vec<_> = infos.iter().flatten().collect();
        let leader = infos.iter().min_by_key(|info| info.doc_freq).unwrap();
        let per_doc = terms
            .len()
            .saturating_mul(std::mem::size_of::<CandidatePosting>() * 2)
            .saturating_add(std::mem::size_of::<DocId>() * 3)
            .saturating_add(std::mem::size_of::<PreparedRow>() * 4);
        estimated_bytes =
            estimated_bytes.saturating_add((leader.doc_freq as usize).saturating_mul(per_doc));
        for info in infos {
            // Reserve whole postings before fetching any term. A follower may
            // choose Whole after the leader is decoded; this conservative reserve
            // prevents a late fallback after reading its metadata or body.
            estimated_bytes =
                estimated_bytes.saturating_add(info.postings_range.len().saturating_mul(2));
            // Decoded skip blocks, selected-range bookkeeping and prefix bytes.
            estimated_bytes = estimated_bytes
                .saturating_add((info.doc_freq as usize / 128 + 1).saturating_mul(256))
                .saturating_add(4096);
        }
        if estimated_bytes > budget {
            return Ok(None);
        }
    }
    if let CandidateShape::Phrase(phrase) = &shape {
        // Preserve the original schema validation even for an empty index.
        phrase.phrase_weight(EnableScoring::disabled_from_searcher(searcher))?;
    }

    let mut rows_by_segment = HashMap::new();
    for (reader, infos) in searcher.segment_readers().iter().zip(segment_infos) {
        let mut rows = Vec::new();
        let infos: Option<Vec<_>> = infos.into_iter().collect();
        if let Some(infos) = infos {
            let option = if phrase {
                IndexRecordOption::WithFreqsAndPositions
            } else {
                IndexRecordOption::WithFreqs
            };
            let mut cursors = Vec::new();
            for (term, info) in terms.iter().zip(&infos) {
                cursors.push(
                    reader
                        .inverted_index(term.field())?
                        .candidate_postings(info, option)?,
                );
            }
            for cursor in &mut cursors {
                cursor.set_read_options(options);
            }
            let leader = infos
                .iter()
                .enumerate()
                .min_by_key(|(_, info)| info.doc_freq)
                .unwrap()
                .0;
            let candidates = cursors[leader].all_docs().await?;
            let mut docs: Vec<_> = candidates.iter().map(|hit| hit.doc).collect();
            let mut leader_hits = Some(candidates);
            let mut all_hits_by_term = Vec::new();
            // Plan each follower's complete range batch once. Smaller decoding
            // batches below must not turn into one remote request wave per batch.
            for (ordinal, cursor) in cursors.iter_mut().enumerate() {
                let hits = if ordinal == leader {
                    let mut hits = leader_hits.take().unwrap();
                    if hits.len() != docs.len() {
                        retain_candidate_hits(&mut hits, &docs);
                    }
                    hits
                } else {
                    cursor.read_docs(&docs).await?
                };
                docs = hits.iter().map(|hit| hit.doc).collect();
                all_hits_by_term.push(hits);
            }
            // Earlier terms may have extra hits eliminated by later terms.
            for hits in &mut all_hits_by_term {
                // The final document set is a subset of every term's hits.
                // Equal lengths therefore mean the lists are already aligned.
                if hits.len() != docs.len() {
                    retain_candidate_hits(hits, &docs);
                }
            }
            if phrase {
                for ((cursor, hits), info) in cursors.iter().zip(&all_hits_by_term).zip(&infos) {
                    if hits.is_empty() {
                        continue;
                    }
                    let block_count = hits.iter().fold(0usize, |sum, hit| {
                        sum.saturating_add((hit.term_freq as usize + 254) / 128)
                    });
                    let raw_bytes = cursor.position_read_bytes(hits)?;
                    let header_bytes = cursor.total_positions().map_or_else(
                        || info.positions_range.len().saturating_mul(9),
                        |total| {
                            usize::try_from(total / 128 + 1)
                                .unwrap_or(usize::MAX)
                                .saturating_mul(9)
                        },
                    );
                    let decoded_batch = hits
                        .chunks(CANDIDATE_BATCH_LEN)
                        .map(|batch| {
                            batch.iter().fold(0usize, |sum, hit| {
                                sum.saturating_add(hit.term_freq as usize)
                            })
                        })
                        .max()
                        .unwrap_or(0)
                        .saturating_mul(16);
                    estimated_bytes = estimated_bytes
                        .saturating_add(raw_bytes.saturating_mul(2))
                        .saturating_add(block_count.saturating_mul(80))
                        .saturating_add(header_bytes)
                        .saturating_add(4096)
                        .saturating_add(decoded_batch);
                    if estimated_bytes > budget {
                        return Ok(None);
                    }
                }
                let mut futures = Vec::with_capacity(cursors.len());
                for (cursor, hits) in cursors.iter_mut().zip(&all_hits_by_term) {
                    futures.push(async move { cursor.prefetch_positions(hits).await }.boxed());
                }
                stream::iter(futures)
                    .buffered(PREPARE_CONCURRENCY)
                    .try_collect::<Vec<_>>()
                    .await?;
            }
            for start in (0..docs.len()).step_by(CANDIDATE_BATCH_LEN) {
                yield_once().await;
                let end = (start + CANDIDATE_BATCH_LEN).min(docs.len());
                let mut hits_by_term: Vec<_> = all_hits_by_term
                    .iter()
                    .map(|hits| hits[start..end].to_vec())
                    .collect();
                match &shape {
                    CandidateShape::Phrase(phrase) => {
                        let mut futures = Vec::with_capacity(cursors.len());
                        for (cursor, hits) in cursors.iter_mut().zip(&mut hits_by_term) {
                            futures.push(async move { cursor.read_positions(hits).await }.boxed());
                        }
                        stream::iter(futures)
                            .buffered(PREPARE_CONCURRENCY)
                            .try_collect::<Vec<_>>()
                            .await?;
                        let postings: Vec<_> = hits_by_term
                            .into_iter()
                            .zip(&infos)
                            .zip(phrase.terms_with_offsets())
                            .map(|((hits, info), (offset, _))| {
                                (
                                    *offset,
                                    BatchPostings {
                                        hits: Arc::from(hits),
                                        cursor: 0,
                                        doc_freq: info.doc_freq,
                                    },
                                )
                            })
                            .collect();
                        // Some(weight) requests full phrase frequency rather than
                        // existence. No scores are computed during preparation.
                        let count_weight =
                            Bm25Weight::new(Explanation::new("phrase count", 1.0), 1.0);
                        let mut scorer = PhraseScorer::new(
                            postings.clone(),
                            Some(count_weight),
                            FieldNormReader::constant(reader.max_doc(), 1),
                            phrase.slop(),
                        );
                        // Preserve the original scorer's existence path too. In
                        // particular, sloppy phrases can take different paths
                        // when computing frequency versus existence only.
                        let mut unscored = PhraseScorer::new(
                            postings,
                            None,
                            FieldNormReader::constant(reader.max_doc(), 1),
                            phrase.slop(),
                        );
                        // Both original semantic modes traverse immutable shared
                        // positions using independent cursors. Merge their sorted
                        // matches directly, without rescanning or sorting rows.
                        while scorer.doc() != TERMINATED || unscored.doc() != TERMINATED {
                            let doc = scorer.doc().min(unscored.doc());
                            let matches_scored = scorer.doc() == doc;
                            let matches_unscored = unscored.doc() == doc;
                            rows.push(PreparedRow {
                                doc,
                                frequencies: [
                                    if matches_scored {
                                        scorer.phrase_count()
                                    } else {
                                        0
                                    },
                                    0,
                                ],
                                matches_unscored,
                            });
                            if matches_scored {
                                scorer.advance();
                            }
                            if matches_unscored {
                                unscored.advance();
                            }
                        }
                    }
                    CandidateShape::Intersection(_) => {
                        for (left, right) in hits_by_term[0].iter().zip(&hits_by_term[1]) {
                            rows.push(PreparedRow {
                                doc: left.doc,
                                frequencies: [left.term_freq, right.term_freq],
                                matches_unscored: true,
                            });
                        }
                    }
                }
            }
        }
        rows_by_segment.insert(reader.segment_id(), Arc::from(rows));
    }
    Ok(Some(PreparedCandidateQuery {
        generation: searcher.generation().clone(),
        terms,
        phrase,
        boost,
        rows: rows_by_segment,
    }))
}

/// Retain the intersection of two sorted unique document lists in linear time.
fn retain_candidate_hits(hits: &mut Vec<CandidatePosting>, docs: &[DocId]) {
    let mut candidate = 0;
    hits.retain(|hit| {
        while candidate < docs.len() && docs[candidate] < hit.doc {
            candidate += 1;
        }
        docs.get(candidate) == Some(&hit.doc)
    });
}

#[derive(Clone)]
struct BatchPostings {
    hits: Arc<[CandidatePosting]>,
    cursor: usize,
    doc_freq: u32,
}

impl DocSet for BatchPostings {
    fn advance(&mut self) -> DocId {
        self.cursor = (self.cursor + 1).min(self.hits.len());
        self.doc()
    }
    fn seek(&mut self, target: DocId) -> DocId {
        self.cursor += self.hits[self.cursor..].partition_point(|hit| hit.doc < target);
        self.doc()
    }
    fn doc(&self) -> DocId {
        self.hits.get(self.cursor).map_or(TERMINATED, |hit| hit.doc)
    }
    fn size_hint(&self) -> u32 {
        self.doc_freq
    }
}

impl Postings for BatchPostings {
    fn term_freq(&self) -> u32 {
        self.hits[self.cursor].term_freq
    }
    fn append_positions_with_offset(&mut self, offset: u32, output: &mut Vec<u32>) {
        output.extend(
            self.hits[self.cursor]
                .positions
                .iter()
                .map(|position| position + offset),
        );
    }
}

impl Query for PreparedCandidateQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        if enable_scoring.searcher().map(Searcher::generation) != Some(&self.generation) {
            return Err(TantivyError::InvalidArgument(
                "prepared query requires its original searcher snapshot".into(),
            ));
        }
        let mut weights = Vec::new();
        if let EnableScoring::Enabled {
            statistics_provider,
            ..
        } = enable_scoring
        {
            if self.phrase {
                weights.push(Bm25Weight::for_terms(statistics_provider, &self.terms)?);
            } else {
                for term in &self.terms {
                    weights.push(Bm25Weight::for_terms(
                        statistics_provider,
                        std::slice::from_ref(term),
                    )?);
                }
            }
        }
        Ok(Box::new(PreparedWeight {
            query: self.clone(),
            weights,
        }))
    }
    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        for term in &self.terms {
            visitor(term, self.phrase);
        }
    }
}

struct PreparedWeight {
    query: PreparedCandidateQuery,
    weights: Vec<Bm25Weight>,
}

impl Weight for PreparedWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        if self.query.generation.segments().get(&reader.segment_id())
            != Some(&reader.delete_opstamp())
        {
            return Err(TantivyError::InvalidArgument(
                "prepared query segment snapshot mismatch".into(),
            ));
        }
        let rows = self
            .query
            .rows
            .get(&reader.segment_id())
            .ok_or_else(|| {
                TantivyError::InvalidArgument("prepared query segment is missing".into())
            })?
            .clone();
        let mut norms = Vec::new();
        for term in self.query.terms.iter().take(self.weights.len()) {
            norms.push(fieldnorm_reader(reader, term.field())?);
        }
        let mut scorer = PreparedScorer {
            rows,
            cursor: 0,
            norms,
            weights: self
                .weights
                .iter()
                .map(|weight| weight.boost_by(boost * self.query.boost))
                .collect(),
        };
        scorer.seek(0);
        Ok(Box::new(scorer))
    }
    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) != doc {
            return Err(super::explanation::does_not_match(doc));
        }
        Ok(Explanation::new("Prepared candidate query", scorer.score()))
    }
}

fn fieldnorm_reader(reader: &SegmentReader, field: Field) -> crate::Result<FieldNormReader> {
    Ok(reader
        .fieldnorms_readers()
        .get_field(field)?
        .unwrap_or_else(|| FieldNormReader::constant(reader.max_doc(), 1)))
}

struct PreparedScorer {
    rows: Arc<[PreparedRow]>,
    cursor: usize,
    norms: Vec<FieldNormReader>,
    weights: Vec<Bm25Weight>,
}

impl DocSet for PreparedScorer {
    fn advance(&mut self) -> DocId {
        self.cursor = (self.cursor + 1).min(self.rows.len());
        self.skip_nonmatches();
        self.doc()
    }
    fn seek(&mut self, target: DocId) -> DocId {
        self.cursor += self.rows[self.cursor..].partition_point(|row| row.doc < target);
        self.skip_nonmatches();
        self.doc()
    }
    fn doc(&self) -> DocId {
        self.rows.get(self.cursor).map_or(TERMINATED, |row| row.doc)
    }
    fn size_hint(&self) -> u32 {
        self.rows.len() as u32
    }
}

impl PreparedScorer {
    fn skip_nonmatches(&mut self) {
        while let Some(row) = self.rows.get(self.cursor) {
            let matches = if self.weights.is_empty() {
                row.matches_unscored
            } else {
                row.frequencies[0] > 0
            };
            if matches {
                break;
            }
            self.cursor += 1;
        }
    }
}

impl Scorer for PreparedScorer {
    fn score(&mut self) -> Score {
        let row = &self.rows[self.cursor];
        if self.weights.is_empty() {
            return 1.0;
        }
        let left = self.weights[0].score(self.norms[0].fieldnorm_id(row.doc), row.frequencies[0]);
        if self.weights.len() == 1 {
            left
        } else {
            left + self.weights[1].score(self.norms[1].fieldnorm_id(row.doc), row.frequencies[1])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::tests::TEST_COLLECTOR_WITH_SCORE;
    use crate::collector::{Count, TopDocs};
    use crate::directory::error::{DeleteError, OpenReadError, OpenWriteError};
    use crate::directory::{
        Directory, FileHandle, OwnedBytes, WatchCallback, WatchHandle, WritePtr,
    };
    use crate::merge_policy::NoMergePolicy;
    use crate::schema::{Schema, TEXT};
    use crate::{Index, TantivyDocument};
    use std::io;
    use std::ops::Range;
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Clone, Debug)]
    struct GuardedDirectory {
        inner: Arc<dyn Directory>,
        forbid_auxiliary_reads: Arc<AtomicBool>,
        guard_terms: bool,
    }

    #[derive(Debug)]
    struct GuardedHandle {
        inner: Arc<dyn FileHandle>,
        forbid: Arc<AtomicBool>,
    }

    impl crate::HasLen for GuardedHandle {
        fn len(&self) -> usize {
            self.inner.len()
        }
    }

    impl FileHandle for GuardedHandle {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            if self.forbid.load(Ordering::SeqCst) {
                return Err(io::Error::other("auxiliary field read during preparation"));
            }
            self.inner.read_bytes(range)
        }
        fn read_bytes_async<'a, 'b>(
            &'a self,
            range: Range<usize>,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = io::Result<OwnedBytes>> + Send + 'b>>
        where
            'a: 'b,
            Self: 'b,
        {
            Box::pin(async move {
                if self.forbid.load(Ordering::SeqCst) {
                    return Err(io::Error::other("auxiliary field read during preparation"));
                }
                self.inner.read_bytes_async(range).await
            })
        }
    }

    impl Directory for GuardedDirectory {
        fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
            let inner = self.inner.get_file_handle(path)?;
            if matches!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("fieldnorm" | "fast")
            ) || (self.guard_terms
                && matches!(
                    path.extension().and_then(|ext| ext.to_str()),
                    Some("idx" | "pos")
                ))
            {
                Ok(Arc::new(GuardedHandle {
                    inner,
                    forbid: self.forbid_auxiliary_reads.clone(),
                }))
            } else {
                Ok(inner)
            }
        }
        fn delete(&self, path: &Path) -> Result<(), DeleteError> {
            self.inner.delete(path)
        }
        fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
            self.inner.exists(path)
        }
        fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
            self.inner.open_write(path)
        }
        fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
            self.inner.atomic_read(path)
        }
        fn atomic_write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
            self.inner.atomic_write(path, data)
        }
        fn sync_directory(&self) -> io::Result<()> {
            self.inner.sync_directory()
        }
        fn watch(&self, callback: WatchCallback) -> crate::Result<WatchHandle> {
            self.inner.watch(callback)
        }
    }

    fn fixture() -> crate::Result<(Index, Field)> {
        let mut schema = Schema::builder();
        let field = schema.add_text_field("text", TEXT);
        let index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_for_tests::<TantivyDocument>()?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for batch in [
            vec!["a b a b", "a x b", "b a", "a a b", "a b remove"],
            vec!["a b", "b a b a", "a x x b", "a a", "b b"],
        ] {
            for text in batch {
                writer.add_document(doc!(field => text))?;
            }
            writer.commit()?;
        }
        writer.delete_term(Term::from_field_text(field, "remove"));
        writer.commit()?;
        Ok((index, field))
    }

    fn check(searcher: &Searcher, query: &dyn Query) -> crate::Result<()> {
        assert!(supports_candidate_query(query));
        let prepared =
            futures::executor::block_on(prepare_candidate_query(searcher, query))?.unwrap();
        for limit in [1, 2, 20] {
            let collector = TopDocs::with_limit(limit).order_by_score();
            assert_eq!(
                searcher.search(query, &collector)?,
                searcher.search(&prepared, &collector)?
            );
        }
        assert_eq!(
            searcher.search(query, &Count)?,
            searcher.search(&prepared, &Count)?
        );
        let original_fruit = searcher.search(query, &TEST_COLLECTOR_WITH_SCORE)?;
        let prepared_fruit = searcher.search(&prepared, &TEST_COLLECTOR_WITH_SCORE)?;
        assert_eq!(original_fruit.docs(), prepared_fruit.docs());
        assert_eq!(
            original_fruit
                .scores()
                .iter()
                .map(|score| score.to_bits())
                .collect::<Vec<_>>(),
            prepared_fruit
                .scores()
                .iter()
                .map(|score| score.to_bits())
                .collect::<Vec<_>>()
        );
        // Count alone could hide a changed document set with the same size.
        // Verify each original existence-mode document, independently of scores.
        let original_unscored = query.weight(EnableScoring::disabled_from_searcher(searcher))?;
        let prepared_unscored = prepared.weight(EnableScoring::disabled_from_searcher(searcher))?;
        for reader in searcher.segment_readers() {
            let mut left = original_unscored.scorer(reader, 1.0)?;
            let mut right = prepared_unscored.scorer(reader, 1.0)?;
            while left.doc() != TERMINATED {
                assert_eq!(left.doc(), right.doc());
                left.advance();
                right.advance();
            }
            assert_eq!(right.doc(), TERMINATED);
        }
        // Exercise Weight::scorer as well as TopDocs' pruning route.
        let original = query.weight(EnableScoring::enabled_from_searcher(searcher))?;
        let prepared = prepared.weight(EnableScoring::enabled_from_searcher(searcher))?;
        for reader in searcher.segment_readers() {
            let mut left = original.scorer(reader, 1.7)?;
            let mut right = prepared.scorer(reader, 1.7)?;
            while left.doc() != TERMINATED {
                assert_eq!(left.doc(), right.doc());
                assert_eq!(left.score().to_bits(), right.score().to_bits());
                left.advance();
                right.advance();
            }
            assert_eq!(right.doc(), TERMINATED);
        }
        Ok(())
    }

    #[test]
    fn prepared_phrase_preserves_slop_offsets_repetition_and_boost() -> crate::Result<()> {
        let (index, field) = fixture()?;
        let searcher = index.reader()?.searcher();
        for (words, offsets) in [
            (vec!["a", "b"], vec![0, 1]),
            (vec!["a", "a"], vec![0, 1]),
            (vec!["a", "b", "a"], vec![0, 1, 2]),
            (vec!["a", "b"], vec![0, 2]),
            (vec!["a", "missing"], vec![0, 1]),
        ] {
            for slop in [0, 1, 2, 4] {
                let terms = offsets
                    .iter()
                    .zip(&words)
                    .map(|(offset, word)| (*offset, Term::from_field_text(field, word)))
                    .collect();
                let phrase = PhraseQuery::new_with_offset_and_slop(terms, slop);
                check(&searcher, &phrase)?;
                check(&searcher, &BoostQuery::new(Box::new(phrase), 2.3))?;
            }
        }
        Ok(())
    }

    #[test]
    fn prepared_two_term_intersection_preserves_scores() -> crate::Result<()> {
        let (index, field) = fixture()?;
        let searcher = index.reader()?.searcher();
        for words in [["a", "b"], ["a", "a"], ["a", "missing"]] {
            let query = BooleanQuery::intersection(
                words
                    .into_iter()
                    .map(|word| {
                        Box::new(TermQuery::new(
                            Term::from_field_text(field, word),
                            IndexRecordOption::WithFreqs,
                        )) as Box<dyn Query>
                    })
                    .collect(),
            );
            check(&searcher, &query)?;
            check(&searcher, &BoostQuery::new(Box::new(query), 2.3))?;
        }
        Ok(())
    }

    #[test]
    fn fast_only_numeric_intersection_keeps_original_fallback() -> crate::Result<()> {
        let mut schema = Schema::builder();
        let left = schema.add_u64_field("left", crate::schema::FAST);
        let right = schema.add_u64_field("right", crate::schema::FAST);
        let index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_for_tests::<TantivyDocument>()?;
        for (a, b) in [(7u64, 7u64), (7, 8), (8, 7)] {
            writer.add_document(doc!(left => a, right => b))?;
        }
        writer.commit()?;
        let searcher = index.reader()?.searcher();
        let query = BooleanQuery::intersection(vec![
            Box::new(TermQuery::new(
                Term::from_field_u64(left, 7),
                IndexRecordOption::WithFreqs,
            )),
            Box::new(TermQuery::new(
                Term::from_field_u64(right, 7),
                IndexRecordOption::WithFreqs,
            )),
        ]);
        for query in [
            Box::new(query.clone()) as Box<dyn Query>,
            Box::new(BoostQuery::new(Box::new(query), 2.0)),
        ] {
            assert!(supports_candidate_query(&*query));
            assert!(
                futures::executor::block_on(prepare_candidate_query(&searcher, &*query))?.is_none()
            );
            assert_eq!(searcher.search(&*query, &Count)?, 1);
            assert!(searcher
                .search(&*query, &TopDocs::with_limit(1).order_by_score())
                .is_err());
        }
        Ok(())
    }

    #[test]
    fn preparation_proves_empty_results_without_auxiliary_fields() -> crate::Result<()> {
        let mut schema = Schema::builder();
        let text = schema.add_text_field("text", TEXT);
        let row_id = schema.add_u64_field("row_id", crate::schema::FAST);
        let directory = crate::directory::RamDirectory::create();
        let index = Index::create(directory.clone(), schema.build(), Default::default())?;
        let mut writer = index.writer_for_tests::<TantivyDocument>()?;
        for (id, value) in ["a x b", "onlyleft", "onlyright"].into_iter().enumerate() {
            writer.add_document(doc!(text => value, row_id => id as u64))?;
        }
        writer.commit()?;
        let forbid = Arc::new(AtomicBool::new(false));
        let guarded = GuardedDirectory {
            inner: Arc::new(directory),
            forbid_auxiliary_reads: forbid.clone(),
            guard_terms: false,
        };
        let guarded_index = Index::open(Box::new(guarded) as Box<dyn Directory>)?;
        let searcher = guarded_index.reader()?.searcher();
        forbid.store(true, Ordering::SeqCst);
        let term = |value| Term::from_field_text(text, value);
        let queries: Vec<Box<dyn Query>> = vec![
            Box::new(BooleanQuery::intersection(vec![
                Box::new(TermQuery::new(
                    term("onlyleft"),
                    IndexRecordOption::WithFreqs,
                )),
                Box::new(TermQuery::new(
                    term("onlyright"),
                    IndexRecordOption::WithFreqs,
                )),
            ])),
            Box::new(PhraseQuery::new(vec![term("a"), term("b")])),
            Box::new(PhraseQuery::new(vec![term("a"), term("missing")])),
        ];
        for query in queries {
            let prepared =
                futures::executor::block_on(prepare_candidate_query(&searcher, &*query))?.unwrap();
            assert!(prepared.is_empty());
        }
        let mut sloppy = PhraseQuery::new(vec![term("a"), term("b")]);
        sloppy.set_slop(1);
        let prepared =
            futures::executor::block_on(prepare_candidate_query(&searcher, &sloppy))?.unwrap();
        assert!(!prepared.is_empty());
        // Establish that the guard would catch the later dependencies, rather
        // than merely observing an already-cached field reader.
        for reader in searcher.segment_readers() {
            assert!(reader.fieldnorms_readers().get_field(text).is_err());
            assert!(reader.fast_fields().u64("row_id").is_err());
        }
        Ok(())
    }

    #[test]
    fn emptiness_preserves_both_sloppy_phrase_modes() -> crate::Result<()> {
        let mut saw_mode_difference = false;
        for texts in [
            vec![
                "a b a b", "a x b", "b a", "a a b", "b a b a", "a x x b", "a a",
            ],
            vec!["a x b"],
        ] {
            let index = crate::query::phrase_query::tests::create_index(&texts)?;
            let text = index.schema().get_field("text")?;
            let searcher = index.reader()?.searcher();
            for words in [&["a", "b"][..], &["a", "a"], &["a", "b", "a"]] {
                for slop in [0, 1, 2, 4] {
                    let query = PhraseQuery::new_with_offset_and_slop(
                        words
                            .iter()
                            .enumerate()
                            .map(|(offset, word)| (offset, Term::from_field_text(text, word)))
                            .collect(),
                        slop,
                    );
                    let scored = searcher
                        .search(&query, &TEST_COLLECTOR_WITH_SCORE)?
                        .docs()
                        .len();
                    let unscored = searcher.search(&query, &Count)?;
                    saw_mode_difference |= scored != unscored;
                    let prepared =
                        futures::executor::block_on(prepare_candidate_query(&searcher, &query))?
                            .unwrap();
                    assert_eq!(
                        prepared.is_empty(),
                        scored == 0 && unscored == 0,
                        "{texts:?}, {words:?}, slop={slop}"
                    );
                }
            }
        }
        assert!(
            saw_mode_difference,
            "fixture must exercise original mode divergence"
        );
        Ok(())
    }

    #[test]
    fn preparation_budget_refuses_before_posting_and_position_reads() -> crate::Result<()> {
        let mut schema = Schema::builder();
        let field = schema.add_text_field("text", TEXT);
        let directory = crate::directory::RamDirectory::create();
        let index = Index::create(directory.clone(), schema.build(), Default::default())?;
        let mut writer = index.writer_for_tests::<TantivyDocument>()?;
        for _ in 0..300 {
            writer.add_document(doc!(field => "a b a b"))?;
        }
        writer.commit()?;
        let forbid = Arc::new(AtomicBool::new(false));
        let guarded = GuardedDirectory {
            inner: Arc::new(directory),
            forbid_auxiliary_reads: forbid.clone(),
            guard_terms: true,
        };
        let index = Index::open(Box::new(guarded) as Box<dyn Directory>)?;
        let searcher = index.reader()?.searcher();
        // Initialize per-field readers before guarding term payloads. Term lookup
        // itself remains permitted, since its ranges are needed for preflight.
        for reader in searcher.segment_readers() {
            reader.inverted_index(field)?;
        }
        forbid.store(true, Ordering::SeqCst);
        let query = PhraseQuery::new(vec![
            Term::from_field_text(field, "a"),
            Term::from_field_text(field, "b"),
        ]);
        assert!(futures::executor::block_on(prepare_with_budget(&searcher, &query, 1))?.is_none());
        assert!(
            futures::executor::block_on(prepare_candidate_query(&searcher, &query)).is_err(),
            "guard must catch payload reads when the budget admits preparation"
        );
        Ok(())
    }

    #[test]
    fn preparation_is_send_yields_and_falls_back_without_truncation() -> crate::Result<()> {
        fn assert_send<T: Send>(_: T) {}
        let (index, field) = fixture()?;
        let searcher = index.reader()?.searcher();
        let phrase = PhraseQuery::new(vec![
            Term::from_field_text(field, "a"),
            Term::from_field_text(field, "b"),
        ]);
        assert_send(prepare_candidate_query(&searcher, &phrase));
        let original_count = searcher.search(&phrase, &Count)?;
        assert!(futures::executor::block_on(prepare_with_budget(&searcher, &phrase, 1))?.is_none());
        assert_eq!(searcher.search(&phrase, &Count)?, original_count);
        // In-memory IO is ready immediately. Pending demonstrates an explicit
        // cooperative boundary and permits cancellation before preparation ends.
        assert!(prepare_candidate_query(&searcher, &phrase)
            .now_or_never()
            .is_none());
        let prepared =
            futures::executor::block_on(prepare_candidate_query(&searcher, &phrase))?.unwrap();
        assert_eq!(searcher.search(&prepared, &Count)?, original_count);
        Ok(())
    }

    #[test]
    fn prepared_phrase_crosses_candidate_and_position_blocks() -> crate::Result<()> {
        let mut schema = Schema::builder();
        let field = schema.add_text_field("text", TEXT);
        let index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_for_tests::<TantivyDocument>()?;
        for id in 0..900 {
            let text = if id == 512 {
                "a b ".repeat(350)
            } else if id % 3 == 0 {
                "a x b".to_string()
            } else {
                "a b a b".to_string()
            };
            writer.add_document(doc!(field => text))?;
        }
        writer.commit()?;
        let searcher = index.reader()?.searcher();
        for slop in [0, 1, 2] {
            for words in [&["a", "b"][..], &["a", "a"], &["a", "b", "a"]] {
                let phrase = PhraseQuery::new_with_offset_and_slop(
                    words
                        .iter()
                        .enumerate()
                        .map(|(offset, word)| (offset, Term::from_field_text(field, word)))
                        .collect(),
                    slop,
                );
                check(&searcher, &phrase)?;
            }
        }
        Ok(())
    }

    #[test]
    fn unsupported_shapes_fall_back_and_other_snapshots_fail() -> crate::Result<()> {
        let (index, field) = fixture()?;
        let term = || {
            Box::new(TermQuery::new(
                Term::from_field_text(field, "a"),
                IndexRecordOption::WithFreqs,
            )) as Box<dyn Query>
        };
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let unsupported: Vec<Box<dyn Query>> = vec![
            term(),
            Box::new(BooleanQuery::intersection(vec![term(), term(), term()])),
            Box::new(BooleanQuery::union(vec![term(), term()])),
            Box::new(BoostQuery::new(
                Box::new(BoostQuery::new(
                    Box::new(BooleanQuery::intersection(vec![term(), term()])),
                    2.0,
                )),
                3.0,
            )),
        ];
        for query in unsupported {
            assert!(!supports_candidate_query(&*query));
            assert!(
                futures::executor::block_on(prepare_candidate_query(&searcher, &*query))?.is_none()
            );
        }
        let query = BooleanQuery::intersection(vec![term(), term()]);
        let prepared =
            futures::executor::block_on(prepare_candidate_query(&searcher, &query))?.unwrap();
        let empty_phrase = PhraseQuery::new(vec![
            Term::from_field_text(field, "a"),
            Term::from_field_text(field, "missing"),
        ]);
        let empty = futures::executor::block_on(prepare_candidate_query(&searcher, &empty_phrase))?
            .unwrap();
        assert!(empty.is_empty());
        let mut writer = index.writer_for_tests::<TantivyDocument>()?;
        writer.add_document(doc!(field => "a b"))?;
        writer.commit()?;
        reader.reload()?;
        assert!(reader.searcher().search(&prepared, &Count).is_err());
        assert!(reader.searcher().search(&empty, &Count).is_err());
        Ok(())
    }
}
