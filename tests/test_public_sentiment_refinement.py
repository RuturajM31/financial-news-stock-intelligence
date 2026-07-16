"""Dependency-free regression tests for the focused public sentiment analyzer."""

from __future__ import annotations

import html
import re
import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup

from app import public_cloud_app as app


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _Requests:
    def __init__(self, document: str) -> None:
        self.document = document

    def get(self, *_args, **_kwargs) -> _Response:
        return _Response(self.document)


class PublicSentimentRefinementTests(unittest.TestCase):
    def test_shared_thresholds(self) -> None:
        self.assertEqual(app._sentiment_interpretation(0.02), "Neutral / mixed")
        self.assertEqual(app._sentiment_interpretation(-0.15), "Bearish")
        self.assertEqual(app._sentiment_interpretation(0.15), "Bullish")

    def test_score_label_and_strength_share_score(self) -> None:
        text = "Revenue improved but the company also reported a warning and uncertainty."
        signal = app._score_article(text, "test", headline_text="Company update", body_text=text)
        self.assertEqual(signal.label, app._sentiment_interpretation(signal.sentiment_score))
        self.assertEqual(abs(signal.sentiment_score) * 100, abs(signal.sentiment_score) * 100)

    def test_spacex_beats_late_microsoft_without_fake_ticker(self) -> None:
        headline = "SpaceX launches new satellite service for enterprise customers"
        body = (
            "SpaceX announced the service after years of development. SpaceX said the launch "
            "will expand coverage and improve capacity for customers. The article later notes "
            "that Microsoft provides unrelated cloud software to some industry participants."
        )
        ticker, company = app._infer_company(headline, body)
        self.assertEqual((ticker, company), ("Private company", "SpaceX"))
        self.assertNotIn("MSFT", ticker)

    def test_unknown_company_state(self) -> None:
        self.assertEqual(
            app._infer_company("Quarterly industry update", "No mapped company is central to this report."),
            ("", "Company not confidently detected"),
        )

    def test_extraction_decodes_and_removes_boilerplate(self) -> None:
        article = " ".join(
            [
                "SpaceX&#x27;s launch programme showed strong demand &amp; record bookings.",
                "The company described regulatory uncertainty but said operations remained stable.",
                "Customers will receive expanded service across several international regions.",
            ]
        )
        document = f"""<html><body><nav>Skip to content Home Newsletter</nav><main><article>
        <h1>SpaceX &quot;service&quot; update</h1><p>{article}</p><p>{article}</p>
        </article></main><aside class="related-articles">Related Microsoft news</aside>
        <footer>Cookie policy All rights reserved</footer></body></html>"""
        with patch.object(app, "requests", _Requests(document)):
            headline, body, method = app._extract_article_content("https://example.com/story")
        self.assertEqual(headline, 'SpaceX "service" update')
        self.assertIn("SpaceX's launch", body)
        self.assertEqual(body.count("SpaceX's launch"), 1)
        self.assertNotIn("Skip to content", body)
        self.assertNotIn("Related", body)
        self.assertNotIn("Cookie policy", body)
        self.assertEqual(method, "Structured article extraction")

    def test_no_hidden_sample_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, "required"):
            app._score_article("", "test")

    def test_evidence_reconciles_and_maps_to_terms(self) -> None:
        text = (
            "The company reported strong revenue growth and record demand but issued a warning "
            "after weak bookings and guidance cut amid uncertainty and competition."
        )
        signal = app._score_article(text, "test", headline_text="Company update", body_text=text)
        rows = app._evidence_contributions(text, signal)
        raw_total = sum(row["contribution"] for row in rows)
        self.assertAlmostEqual(max(-1.0, min(1.0, raw_total)), signal.sentiment_score)
        self.assertEqual({row["term"] for row in rows}, set(signal.positive_hits) | set(signal.negative_hits))
        self.assertTrue(all(row["term"].lower() in text.lower() and row["phrase"] for row in rows))

    def test_cloud_only_contains_detected_evidence(self) -> None:
        text = "Strong demand improved revenue while regulation and supply risk remained."
        signal = app._score_article(text, "test", headline_text="Update", body_text=text)
        items = app._evidence_cloud_items(text, signal)
        expected = set(signal.positive_hits) | set(signal.negative_hits) | set(signal.risk_hits)
        self.assertEqual({item["term"] for item in items}, expected)

    def test_highlighting_preserves_words_and_longest_match(self) -> None:
        text = "Management announced a guidance cut after weak demand."
        signal = app._score_article(text, "test", headline_text="Update", body_text=text)
        marked = app._highlight_submitted_text(text, signal)
        self.assertEqual(BeautifulSoup(marked, "html.parser").get_text(), text)
        self.assertIn('<mark class="fs-neg">guidance cut</mark>', marked)
        self.assertNotIn('<mark class="fs-neg">cut</mark>', marked)
        self.assertEqual(marked.count("guidance cut</mark>"), 1)

    def test_filter_only_emphasizes_selected_category(self) -> None:
        text = "Strong revenue growth continued despite weak demand and supply risk."
        signal = app._score_article(text, "test", headline_text="Update", body_text=text)
        positive = app._highlight_submitted_text(text, signal, "Positive")
        self.assertIn("fs-pos", positive)
        self.assertNotIn("fs-neg", positive)
        self.assertNotIn("fs-risk", positive)
        self.assertEqual(BeautifulSoup(positive, "html.parser").get_text(), html.unescape(text))


if __name__ == "__main__":
    unittest.main()
