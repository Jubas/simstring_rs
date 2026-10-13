//! `HashDb::shrink_to_fit` may only release capacity. These tests pin the index and every
//! search answer, bit for bit, before and after shrinking, and after inserting into a
//! shrunk database. Search correctness itself is checked against brute force in
//! `test_search_corpus.rs`, which also runs on a shrunk database.

use lasso::Spur;
use simstring_rust::{
    CharacterNgrams, Cosine, Database, Dice, ExactMatch, FeatureExtractor, HashDb, Jaccard,
    Measure, Overlap, Searcher, WordNgrams,
};
use std::sync::Arc;

type Postings = Vec<(Spur, usize, Vec<usize>)>;

/// Everything search can observe through the `Database` trait.
#[derive(Debug, PartialEq)]
struct Snapshot {
    strings: Vec<String>,
    features: Vec<Vec<Spur>>,
    feature_sizes: Vec<usize>,
    max_feature_len: usize,
    total_strings: usize,
    interned: Vec<(Spur, String)>,
    lookups: Postings,
    visits: Vec<((usize, usize), Postings)>,
}

fn snapshot(db: &HashDb) -> Snapshot {
    let total_strings = db.total_strings();
    let interned: Vec<_> = db
        .interner()
        .iter()
        .map(|(key, text)| (key, text.to_owned()))
        .collect();
    let sizes = db.feature_sizes().to_vec();
    let max = db.max_feature_len();

    // Absent sizes are probed too: they must stay absent.
    let mut probe = sizes.clone();
    probe.extend([0, max + 1, usize::MAX]);
    let mut lookups = Vec::new();
    for &(feature, _) in &interned {
        for &size in &probe {
            if let Some(ids) = db.lookup_strings(size, feature) {
                lookups.push((feature, size, ids.to_vec()));
            }
        }
    }

    // Full, empty, inverted and partial size windows.
    let mut windows = vec![(0, usize::MAX), (0, 0), (max + 1, usize::MAX), (max, 1)];
    for (i, &lo) in sizes.iter().enumerate().step_by(4) {
        let hi = sizes[(i + 3).min(sizes.len() - 1)];
        windows.extend([(lo, hi), (lo + 1, hi), (lo, lo)]);
    }
    let visits = windows
        .into_iter()
        .map(|(min, max)| {
            let mut seen = Vec::new();
            for &(feature, _) in &interned {
                db.visit_postings(feature, min, max, &mut |size, ids| {
                    seen.push((feature, size, ids.to_vec()));
                });
            }
            ((min, max), seen)
        })
        .collect();

    Snapshot {
        strings: (0..total_strings)
            .map(|id| db.get_string(id).unwrap().to_owned())
            .collect(),
        features: (0..total_strings)
            .map(|id| db.get_features(id).unwrap().to_vec())
            .collect(),
        feature_sizes: sizes,
        max_feature_len: max,
        total_strings,
        interned,
        lookups,
        visits,
    }
}

fn build(extractor: &Arc<dyn FeatureExtractor>, corpus: &[String]) -> HashDb {
    let mut db = HashDb::new(Arc::clone(extractor));
    for text in corpus {
        db.insert(text.clone());
    }
    db
}

fn read_lines(file: &str, limit: usize) -> Vec<String> {
    let path = format!("{}/benches/data/{file}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .take(limit)
        .map(str::to_owned)
        .collect()
}

/// Inputs that stress empty feature sets, duplicates, multi-byte text and long strings.
fn edge_cases() -> Vec<String> {
    let mut texts: Vec<String> = [
        "",
        "a",
        "aa",
        "aaaa",
        "aaaa",
        "東京🍕",
        "café",
        "a b a b",
        "a b c",
        "the quick brown fox jumps over the lazy dog",
    ]
    .map(String::from)
    .into();
    texts.push("ab".repeat(40));
    texts.push("x".repeat(200));
    texts
}

fn corpus() -> Vec<String> {
    let mut corpus = read_lines("company_names.txt", 2500);
    corpus.extend(edge_cases());
    corpus
}

/// Indexed strings, near misses of indexed strings and strings that are not indexed.
fn queries(corpus: &[String], count: usize) -> Vec<String> {
    let mut queries: Vec<String> = corpus
        .iter()
        .step_by(corpus.len() / count)
        .cloned()
        .collect();
    let near: Vec<String> = queries
        .iter()
        .step_by(3)
        .flat_map(|q| {
            let chars: Vec<char> = q.chars().collect();
            let mid = chars.len() / 2;
            let dropped: String = chars[..mid]
                .iter()
                .chain(&chars[(mid + 1).min(chars.len())..])
                .collect();
            [
                dropped,
                q.to_uppercase(),
                q.chars().rev().collect(),
                format!("{q} inc"),
            ]
        })
        .collect();
    queries.extend(near);
    queries.extend(edge_cases());
    queries.extend(["never seen 🦀", "zzzzqqqq", "International"].map(String::from));
    queries
}

/// Thresholds for one query: fixed values plus the exact scores of its hits, where `>=`
/// and tie ordering matter most.
fn thresholds<M: Measure + Copy>(searcher: &Searcher<'_, M>, query: &str) -> Vec<f64> {
    let mut alphas = vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let mut scores: Vec<f64> = searcher
        .ranked_search(query, 0.3)
        .unwrap()
        .into_iter()
        .map(|(_, score)| score)
        .collect();
    scores.dedup();
    alphas.extend(scores.iter().step_by(scores.len().div_ceil(8).max(1)));
    alphas
}

/// Asserts both databases give identical `search` and `ranked_search` answers, scores
/// compared bit for bit. Returns the number of hits so callers can rule out vacuous runs.
fn assert_same_answers<M: Measure + Copy>(
    plain: &HashDb,
    shrunk: &HashDb,
    measure: M,
    queries: &[String],
    alphas: Option<&[f64]>,
    label: &str,
) -> usize {
    let bits = |v: Vec<(&str, f64)>| -> Vec<(String, u64)> {
        v.into_iter()
            .map(|(t, s)| (t.to_owned(), s.to_bits()))
            .collect()
    };
    let plain_searcher = Searcher::new(plain, measure);
    let shrunk_searcher = Searcher::new(shrunk, measure);
    let mut hits = 0;
    for query in queries {
        let alphas = match alphas {
            Some(alphas) => alphas.to_vec(),
            None => thresholds(&plain_searcher, query),
        };
        for alpha in alphas {
            let expected = plain_searcher.ranked_search(query, alpha).unwrap();
            hits += expected.len();
            assert_eq!(
                bits(shrunk_searcher.ranked_search(query, alpha).unwrap()),
                bits(expected),
                "{label}: ranked_search q={query:?} alpha={alpha}"
            );
            assert_eq!(
                shrunk_searcher.search(query, alpha).unwrap(),
                plain_searcher.search(query, alpha).unwrap(),
                "{label}: search q={query:?} alpha={alpha}"
            );
        }
    }
    hits
}

fn assert_all_measures(plain: &HashDb, shrunk: &HashDb, queries: &[String], label: &str) {
    let hits = [
        assert_same_answers(
            plain,
            shrunk,
            Cosine,
            queries,
            None,
            &format!("{label} cosine"),
        ),
        assert_same_answers(plain, shrunk, Dice, queries, None, &format!("{label} dice")),
        assert_same_answers(
            plain,
            shrunk,
            Jaccard,
            queries,
            None,
            &format!("{label} jaccard"),
        ),
        assert_same_answers(
            plain,
            shrunk,
            Overlap,
            queries,
            None,
            &format!("{label} overlap"),
        ),
        assert_same_answers(
            plain,
            shrunk,
            ExactMatch,
            queries,
            None,
            &format!("{label} exact"),
        ),
    ];
    assert!(hits.iter().all(|&h| h > 0), "{label}: vacuous run {hits:?}");
}

fn extractors() -> Vec<(&'static str, Arc<dyn FeatureExtractor>)> {
    vec![
        ("char2", Arc::new(CharacterNgrams::new(2, " "))),
        ("char3", Arc::new(CharacterNgrams::new(3, " "))),
        ("char4", Arc::new(CharacterNgrams::new(4, " "))),
        (
            "char3-unicode-marker",
            Arc::new(CharacterNgrams::new(3, "🍕!")),
        ),
        ("char2-no-marker", Arc::new(CharacterNgrams::new(2, ""))),
        ("word2", Arc::new(WordNgrams::new(2, " ", "$"))),
    ]
}

#[test]
fn index_is_unchanged_by_shrinking() {
    let corpus = corpus();
    for (label, extractor) in extractors() {
        let mut db = build(&extractor, &corpus);
        let before = snapshot(&db);
        assert_eq!(before.total_strings, corpus.len());
        db.shrink_to_fit();
        assert!(snapshot(&db) == before, "{label}: index changed");
        // Shrinking is idempotent.
        db.shrink_to_fit();
        assert!(
            snapshot(&db) == before,
            "{label}: second shrink changed the index"
        );
    }
}

#[test]
fn search_answers_are_unchanged_by_shrinking() {
    let corpus = corpus();
    let queries = queries(&corpus, 12);
    for (label, extractor) in extractors() {
        let plain = build(&extractor, &corpus);
        let mut shrunk = build(&extractor, &corpus);
        assert!(
            snapshot(&shrunk) == snapshot(&plain),
            "{label}: builds differ"
        );
        shrunk.shrink_to_fit();
        assert_all_measures(&plain, &shrunk, &queries, label);
    }
}

#[test]
fn parallel_search_is_unchanged_by_shrinking() {
    // Search fans out over Rayon only from 100_000 strings on; use enough to cross that.
    let corpus = read_lines("eng_google_books_unigram.txt", 120_000);
    assert!(corpus.len() >= 100_000);
    let extractor: Arc<dyn FeatureExtractor> = Arc::new(CharacterNgrams::new(2, " "));
    let plain = build(&extractor, &corpus);
    let mut shrunk = build(&extractor, &corpus);
    shrunk.shrink_to_fit();
    // Long queries span many size groups, so several joins run in parallel.
    let mut queries: Vec<String> = corpus
        .iter()
        .filter(|text| text.chars().count() >= 8)
        .step_by(4000)
        .cloned()
        .collect();
    queries.extend(["international", "abcdefghij", "🦀🦀🦀"].map(String::from));
    let alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let hits = [
        assert_same_answers(&plain, &shrunk, Cosine, &queries, Some(&alphas), "cosine"),
        assert_same_answers(&plain, &shrunk, Jaccard, &queries, Some(&alphas), "jaccard"),
        assert_same_answers(&plain, &shrunk, Overlap, &queries, Some(&alphas), "overlap"),
    ];
    assert!(hits.iter().all(|&h| h > 0), "vacuous run {hits:?}");
}

#[test]
fn inserts_after_shrinking_match_an_unshrunk_build() {
    let corpus = corpus();
    let queries = queries(&corpus, 12);
    let third = corpus.len() / 3;
    for (label, extractor) in extractors() {
        let plain = build(&extractor, &corpus);
        // Shrink between batches; later batches bring new features and sizes, so shrunk
        // buffers must grow again.
        let mut shrunk = HashDb::new(Arc::clone(&extractor));
        for (i, text) in corpus.iter().enumerate() {
            if i == third || i == 2 * third {
                shrunk.shrink_to_fit();
                shrunk.shrink_to_fit();
            }
            shrunk.insert(text.clone());
        }
        assert!(
            snapshot(&shrunk) == snapshot(&plain),
            "{label}: index differs"
        );
        assert_all_measures(&plain, &shrunk, &queries, label);
        shrunk.shrink_to_fit();
        assert!(
            snapshot(&shrunk) == snapshot(&plain),
            "{label}: final shrink"
        );
    }
}

#[test]
fn shrinking_empty_and_cleared_databases() {
    let corpus = corpus();
    let queries = queries(&corpus, 12);
    let extractor: Arc<dyn FeatureExtractor> = Arc::new(CharacterNgrams::new(3, " "));
    let empty = HashDb::new(Arc::clone(&extractor));

    let mut db = HashDb::new(Arc::clone(&extractor));
    db.shrink_to_fit();
    assert!(snapshot(&db) == snapshot(&empty));
    let searcher = Searcher::new(&db, Cosine);
    for query in &queries {
        assert!(searcher.search(query, 0.5).unwrap().is_empty());
    }

    // Fill, clear, shrink and refill: the result must equal a fresh build.
    let plain = build(&extractor, &corpus);
    for text in &corpus {
        db.insert(text.clone());
    }
    db.clear();
    db.shrink_to_fit();
    assert!(snapshot(&db) == snapshot(&empty));
    for text in &corpus {
        db.insert(text.clone());
    }
    db.shrink_to_fit();
    assert!(snapshot(&db) == snapshot(&plain));
    let hits = assert_same_answers(&plain, &db, Cosine, &queries, None, "refilled");
    assert!(hits > 0, "refilled: vacuous run");
}
