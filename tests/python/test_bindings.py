import pytest
from collections import Counter
from pathlib import Path

from simstring_rust.database import HashDb
from simstring_rust.errors import SearchError
from simstring_rust.extractors import CharacterNgrams, WordNgrams, CustomExtractor
from simstring_rust.measures import Cosine, Dice, ExactMatch, Jaccard, Overlap
from simstring_rust.searcher import Searcher
from concurrent.futures import ThreadPoolExecutor


class TestSimstringBindings:
    def setup_method(self):
        self.extractor = CharacterNgrams(n=2, endmarker="$")
        self.db = HashDb(self.extractor)
        # Insert in non-alphabetical order to also test sorting logic
        self.db.insert("apply")
        self.db.insert("apple")
        self.db.insert("banana")
        self.searcher = Searcher(self.db, Cosine())

    def test_db_creation_and_insertion(self):
        # This test requires a clean DB
        db = HashDb(self.extractor)
        assert len(db) == 0

        db.insert("apple")
        db.insert("apply")
        assert len(db) == 2

        db.clear()
        assert len(db) == 0

    def test_db_shrink_to_fit(self):
        expected = self.searcher.ranked_search("appl", 0.3)
        self.db.shrink_to_fit()
        assert self.searcher.ranked_search("appl", 0.3) == expected
        self.db.insert("applet")
        assert self.searcher.search("applet", 1.0) == ["applet"]

    def test_db_shrink_to_fit_preserves_all_answers(self):
        path = Path(__file__).parents[2] / "benches" / "data" / "company_names.txt"
        corpus = path.read_text(encoding="utf-8").splitlines()[:1500]
        corpus += ["", "aaaa", "aaaa", "東京🍕", "café", "ab" * 40]
        queries = corpus[::60] + ["never seen 🦀", "International", "东京", ""]
        queries += [q[::-1] for q in corpus[::150]] + [q.upper() for q in corpus[::150]]
        alphas = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        measures = [Cosine(), Dice(), Jaccard(), Overlap(), ExactMatch()]

        def build(texts):
            db = HashDb(CharacterNgrams(n=3, endmarker=" "))
            for text in texts:
                db.insert(text)
            return db

        def answers(db):
            out = []
            for measure in measures:
                searcher = Searcher(db, measure)
                for query in queries:
                    for alpha in alphas:
                        out.append(
                            (
                                searcher.ranked_search(query, alpha),
                                searcher.search(query, alpha),
                            )
                        )
            return out

        plain = build(corpus)
        expected = answers(plain)
        assert sum(len(ranked) for ranked, _ in expected) > 0

        # Same database before and after shrinking; floats compared exactly.
        shrunk = build(corpus)
        assert answers(shrunk) == expected
        shrunk.shrink_to_fit()
        assert shrunk.strings() == plain.strings()
        assert answers(shrunk) == expected
        shrunk.shrink_to_fit()
        assert answers(shrunk) == expected

        # Inserting into a shrunk database matches a database that was never shrunk.
        half = len(corpus) // 2
        grown = build(corpus[:half])
        grown.shrink_to_fit()
        for text in corpus[half:]:
            grown.insert(text)
        assert grown.strings() == plain.strings()
        assert answers(grown) == expected

    def test_db_strings(self):
        db = HashDb(self.extractor)

        db.insert("apple")
        db.insert("apply")

        db_collection = db.strings()
        assert db_collection == ["apple", "apply"]

    def test_ranked_search_correctness(self):
        # With a threshold of 0.8, only "apple" should be returned.
        results = self.searcher.ranked_search("apple", 0.8)
        assert len(results) == 1
        assert results[0][0] == "apple"
        assert results[0][1] == pytest.approx(1.0)

        # With a lower threshold, "apply" should also be returned.
        results_lower_thresh = self.searcher.ranked_search("apple", 0.6)
        assert len(results_lower_thresh) == 2
        assert results_lower_thresh[0][0] == "apple"
        assert results_lower_thresh[1][0] == "apply"
        assert results_lower_thresh[1][1] == pytest.approx(4 / 6)

    def test_unranked_search_correctness(self):
        # With a threshold of 0.8, only "apple" should be returned.
        results = self.searcher.search("apple", 0.8)
        assert results == ["apple"]

        # With a lower threshold, both are returned, sorted alphabetically.
        results_lower_thresh = self.searcher.search("apple", 0.6)
        assert results_lower_thresh == ["apple", "apply"]

    def test_search_error_on_invalid_threshold(self):
        with pytest.raises(SearchError, match=r"Invalid threshold: 1\.1"):
            self.searcher.search("test", 1.1)

        with pytest.raises(SearchError, match=r"Invalid threshold: 0(\.0)?"):
            self.searcher.search("test", 0.0)

    def test_concurrent_searches(self):
        queries = ["apple", "apply", "unknown 🦀", "banana"] * 100
        expected = [self.searcher.ranked_search(query, 0.6) for query in queries]
        with ThreadPoolExecutor(max_workers=8) as executor:
            actual = list(executor.map(lambda query: self.searcher.ranked_search(query, 0.6), queries))
        assert actual == expected

    def test_unknown_features_and_reinsertion(self):
        assert self.searcher.search("apple🦀", 1.0) == []
        self.db.clear()
        self.db.insert("apple🦀")
        assert self.searcher.ranked_search("apple🦀", 1.0) == [("apple🦀", 1.0)]

    def test_long_inputs_match_python_reference(self):
        # Long strings and many duplicates produce large feature sets and long postings.
        extractor = CharacterNgrams(n=3, endmarker="$")
        db = HashDb(extractor)
        words = ["acme", "acne", "apex", "axle"]
        corpus = [f"{words[i % 4]} company {i % 7} international holdings" for i in range(200)]
        for text in corpus:
            db.insert(text)
        searcher = Searcher(db, Cosine())
        query = set(extractor.apply("acme company 3 international holdings"))
        for alpha in [0.5, 0.8, 0.95, 1.0]:
            expected = {}
            for text in corpus:
                features = set(extractor.apply(text))
                score = len(query & features) / (len(query) * len(features)) ** 0.5
                if score >= alpha - 1e-12:
                    expected.setdefault(text, []).append(score)
            actual = {}
            for text, score in searcher.ranked_search("acme company 3 international holdings", alpha):
                actual.setdefault(text, []).append(score)
            assert actual.keys() == expected.keys()
            for text, scores in actual.items():
                assert scores == pytest.approx(expected[text])

    def test_character_ngram_apply(self):
        extractor = CharacterNgrams(n=2, endmarker="$")
        features = extractor.apply("apple")

        expected = ["$a1", "ap1", "pp1", "pl1", "le1", "e$1"]
        assert Counter(features) == Counter(expected)

    def test_word_ngram_apply(self):
        extractor = WordNgrams(n=2, splitter=" ", padder="#")
        features = extractor.apply("foo bar baz")

        expected = ["# foo1", "foo bar1", "bar baz1", "baz #1"]
        assert Counter(features) == Counter(expected)

    def test_custom_extractor_apply(self):
        class UnigramExtractor:
            def apply(self, text: str):
                return list(text)

        extractor = CustomExtractor(UnigramExtractor())
        features = extractor.apply("foo")

        expected = ["f1", "o1", "o2"]
        assert Counter(features) == Counter(expected)

    def test_custom_extractor_in_db(self):
        class UnigramExtractor:
            def apply(self, text: str):
                return list(text)

        extractor = CustomExtractor(UnigramExtractor())
        db = HashDb(extractor)
        db.insert("foo")
        db.insert("bar")

        searcher = Searcher(db, Cosine())
        results = searcher.search("foo", 0.8)

        assert results == ["foo"]

    def test_custom_extractor_concurrent_queries(self):
        class TokenExtractor:
            def apply(self, text):
                yield from text.split()

        db = HashDb(CustomExtractor(TokenExtractor()))
        db.insert("foo foo bar")
        db.insert("foo bar")
        searcher = Searcher(db, Cosine())
        queries = ["foo foo bar", "foo bar", "foo unknown", "🦀"] * 25
        expected = [searcher.ranked_search(query, 0.5) for query in queries]
        assert searcher.ranked_search("foo unknown", 0.5) == [("foo bar", 0.5)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            actual = list(executor.map(lambda query: searcher.ranked_search(query, 0.5), queries))
        assert actual == expected

    def test_word_ngram_edge_cases(self):
        # Empty string
        extractor = WordNgrams(n=2, splitter=" ", padder="#")
        features = extractor.apply("")
        # With n=2 and 1 padding on each side, we get ["# #"] -> ["# #1"]
        assert features == ["# #1"]

        # String with only separators
        features_sep = extractor.apply("   ")
        assert features_sep == ["# #1"]

        # Different splitter
        extractor_comma = WordNgrams(n=2, splitter=",", padder="#")
        features_comma = extractor_comma.apply("foo,bar")
        expected_comma = ["# foo1", "foo bar1", "bar #1"]
        assert Counter(features_comma) == Counter(expected_comma)

    def test_word_ngrams_in_db(self):
        extractor = WordNgrams(n=2, splitter=" ", padder="#")
        db = HashDb(extractor)
        db.insert("foo bar")
        searcher = Searcher(db, Cosine())
        results = searcher.search("foo bar", 1.0)
        assert results == ["foo bar"]

    def test_invalid_extractor_in_db(self):
        with pytest.raises(TypeError, match="Extractor must be CharacterNgrams, WordNgrams, or CustomExtractor"):
            HashDb("not an extractor")

    def test_ranked_search_error_on_invalid_threshold(self):
        with pytest.raises(SearchError, match=r"Invalid threshold: 1\.1"):
            self.searcher.ranked_search("test", 1.1)
