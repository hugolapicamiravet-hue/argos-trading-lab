"""Deterministic identity and deduplication.

Two properties matter here and both are easy to get wrong:

- Identity must be stable across processes and runs. Python's built-in
  `hash()` is randomised per interpreter by design, so anything keyed on it
  would dedupe correctly once and then produce a different set of ids forever
  after. Every key here is a SHA-256.
- Identity must survive harmless variation. The same story delivered twice with
  a tracking parameter, a trailing slash, or `www` attached is one story, and
  counting it twice would double its weight in the news score.
"""

from __future__ import annotations

import datetime as dt

import pytest
from tests.news.conftest import UTC, make_article

from argos.news import article_key, canonical_url, dedupe_articles
from argos.news.dedupe import article_identity


class TestCanonicalUrl:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("https://example.com/a", "https://example.com/a"),
            ("HTTPS://EXAMPLE.COM/a", "https://example.com/a"),
            # http is upgraded to https: the same article behind both schemes
            # is one article, not two.
            ("http://example.com/a", "https://example.com/a"),
            ("https://example.com:443/a", "https://example.com/a"),
            ("https://example.com:80/a", "https://example.com/a"),
            ("https://example.com/a/", "https://example.com/a"),
            ("https://example.com/a?b=1", "https://example.com/a?b=1"),
        ],
    )
    def test_normalizes_scheme_host_port_and_trailing_slash(self, raw, expected):
        assert canonical_url(raw) == expected

    @pytest.mark.parametrize(
        "param",
        [
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "gclid", "fbclid", "mc_cid", "mc_eid", "ref", "ref_src", "cmpid",
            "smid", "yclid", "igshid", "source", "partner", "trk",
        ],
    )
    def test_strips_known_tracking_parameters(self, param):
        url = f"https://example.com/a?id=7&{param}=xyz&page=2"
        cleaned = canonical_url(url)
        assert param not in cleaned
        assert "id=7" in cleaned and "page=2" in cleaned

    def test_preserves_meaningful_query_parameters(self):
        # `id` and `page` change which article is shown. Stripping them would
        # collapse genuinely different pages into one identity.
        url = "https://example.com/a?id=7&page=2"
        assert canonical_url(url) == "https://example.com/a?id=7&page=2"

    def test_preserves_non_default_port(self):
        assert canonical_url("https://example.com:8443/a") == "https://example.com:8443/a"

    def test_removes_fragment(self):
        assert canonical_url("https://example.com/a#section") == "https://example.com/a"

    def test_normalizes_empty_and_missing(self):
        assert canonical_url(None) is None
        assert canonical_url("") is None
        assert canonical_url("   ") is None

    def test_scheme_less_url_is_treated_as_https(self):
        # Some feeds emit "example.com/story" with no scheme at all.
        assert canonical_url("example.com/story") == "https://example.com/story"

    def test_garbage_does_not_raise(self):
        # A malformed URL is not fatal: identity falls back to the next level of
        # the cascade, which is the fingerprint. Crashing the whole fetch over
        # one bad row would be the wrong trade.
        assert canonical_url("not a url") is not None

    def test_credentials_in_url_are_discarded(self):
        # A feed that embeds credentials must not get them into a stored dataset
        # or a manifest.
        cleaned = canonical_url("https://user:hunter2@example.com/a")
        assert cleaned == "https://example.com/a"
        assert "hunter2" not in (cleaned or "")

    def test_removes_www_prefix(self):
        assert canonical_url("https://www.example.com/a") == "https://example.com/a"

    def test_keeps_subdomains(self):
        assert canonical_url("https://news.example.com/a") == "https://news.example.com/a"


class TestArticleKey:
    """Id prefixes make the cascade level visible in stored data.

    When a dataset is questioned six months later, the key itself should say
    *which* level of the cascade produced the identity: `provider:` means the
    provider's own id, `url:` means a canonical URL, `fp:` means content. Those
    have different failure modes, and a bare `massive:n1` would have hidden it.
    """

    def test_uses_provider_id_when_present(self):
        article = make_article(raw_source_id="abc123")
        assert article_key(article) == "provider:massive:abc123"

    def test_uses_canonical_url_without_provider_id(self):
        article = make_article(raw_source_id=None)
        expected = "url:" + canonical_url(article.url)
        assert article_key(article) == expected

    def test_falls_back_to_fingerprint(self):
        article = make_article(raw_source_id=None, url=None)
        assert article_key(article) == f"fp:{article.fingerprint}"

    def test_key_is_stable_and_hash_free(self):
        article = make_article()
        assert article_key(article) == article_key(article)
        # A salted repr would vary per process; pinning the exact string makes a
        # change to key derivation a visible, deliberate diff rather than a
        # silent reindex of the whole store.
        assert article_key(article) == "provider:massive:n1"

    def test_article_id_mirrors_the_key(self):
        article = make_article()
        assert article.article_id == article_key(article)


class TestArticleIdentity:
    def test_extracts_provider_id(self):
        identity = article_identity({"id": "abc", "title": "x"}, source="massive")
        assert identity == "provider:massive:abc"

    @pytest.mark.parametrize("key", ["id", "article_id", "uuid", "guid"])
    def test_accepts_alternate_id_keys(self, key):
        assert article_identity({key: "abc"}, source="massive") == "provider:massive:abc"

    def test_falls_back_to_url(self):
        identity = article_identity(
            {"title": "x", "article_url": "https://example.com/a?utm_source=t"},
            source="massive",
        )
        assert identity == "url:https://example.com/a"

    def test_falls_back_to_fingerprint(self):
        identity = article_identity({"title": "Story", "summary": "Body"}, source="massive")
        assert identity.startswith("fp:")

    def test_requires_source(self):
        with pytest.raises(ValueError, match="must name the provider"):
            article_identity({"id": "a"}, source="")

    def test_non_dict_payload_is_a_type_error(self):
        # Refusing a non-dict is correct: guessing an identity for something
        # that is not a record would invent data rather than reject it.
        for junk in (None, [], "string", 42):
            with pytest.raises(TypeError, match="must be a dict"):
                article_identity(junk, source="massive")

    def test_empty_id_is_not_trusted(self):
        # An empty id is worse than none: it would dedupe every such record
        # into a single row.
        identity = article_identity(
            {"id": "   ", "title": "Distinct story"}, source="massive"
        )
        assert identity.startswith("fp:")

    def test_missing_id_fields_reach_fingerprint(self):
        assert article_identity({}, source="massive").startswith("fp:")


class TestDedupeArticles:
    def test_keeps_first_occurrence(self):
        first = make_article(article_id="provider:massive:n1", raw_source_id="n1")
        second = make_article(article_id="provider:massive:n1", raw_source_id="n1")
        kept, duplicate_ids = dedupe_articles([first, second])
        assert kept == (first,)
        assert duplicate_ids == ("provider:massive:n1",)

    def test_reports_one_entry_per_dropped_row(self):
        # The count is the useful number here: "3 rows dropped" tells an
        # operator the dedupe actually engaged, while a deduplicated list of
        # identities would report 1 and hide the scale.
        articles = [make_article(raw_source_id="n1") for _ in range(4)]
        kept, duplicate_ids = dedupe_articles(articles)
        assert len(kept) == 1
        assert duplicate_ids == ("provider:massive:n1",) * 3

    def test_distinct_articles_are_all_kept(self):
        articles = [
            make_article(raw_source_id="n1"),
            make_article(raw_source_id="n2"),
            make_article(raw_source_id="n3"),
        ]
        kept, duplicates = dedupe_articles(articles)
        assert len(kept) == 3
        assert duplicates == ()

    def test_output_is_sorted_by_publication_time_then_id(self):
        # Output order must not depend on input order, otherwise two runs over
        # the same data in different fetch orders produce different files.
        articles = [
            make_article(raw_source_id="n3", published_at=dt.datetime(2025, 7, 3, tzinfo=UTC)),
            make_article(raw_source_id="n1", published_at=dt.datetime(2025, 7, 1, tzinfo=UTC)),
            make_article(raw_source_id="n2", published_at=dt.datetime(2025, 7, 2, tzinfo=UTC)),
        ]
        kept, _ = dedupe_articles(articles)
        assert [a.raw_source_id for a in kept] == ["n1", "n2", "n3"]
        assert [a.raw_source_id for a in dedupe_articles(list(reversed(articles)))[0]] == [
            "n1", "n2", "n3",
        ]

    def test_earliest_published_copy_wins(self):
        # A syndicated story reaches us twice with slightly different
        # timestamps. The first appearance is the one closest to what the
        # market saw, so the later copy is dropped.
        early = make_article(
            raw_source_id="n1", published_at=dt.datetime(2025, 7, 31, 20, 0, tzinfo=UTC)
        )
        late = make_article(
            raw_source_id="n1", published_at=dt.datetime(2025, 7, 31, 22, 0, tzinfo=UTC)
        )
        kept, dropped = dedupe_articles([late, early])
        assert kept == (early,)
        assert dropped == (late.article_id,)

    def test_earliest_wins_regardless_of_input_order(self):
        early = make_article(
            raw_source_id="n1", published_at=dt.datetime(2025, 7, 31, 20, 0, tzinfo=UTC)
        )
        late = make_article(
            raw_source_id="n1", published_at=dt.datetime(2025, 7, 31, 22, 0, tzinfo=UTC)
        )
        assert dedupe_articles([late, early]) == dedupe_articles([early, late])

    def test_empty_input(self):
        assert dedupe_articles([]) == ((), ())

    def test_generator_input_is_accepted(self):
        kept, _ = dedupe_articles(make_article(raw_source_id=f"n{i}") for i in range(3))
        assert len(kept) == 3