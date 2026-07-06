"""Text-extraction error tests."""

import csv

import pytest

from financial_news_intelligence.data import text_extraction


# ============================================================
# 1. EMPTY TEXT
# ============================================================

def test_empty_text_is_rejected() -> None:
    """Empty pasted text should raise a clear error."""

    # Prepare text containing spaces but no readable words.
    empty_text = "   "

    # Confirm the cleaner rejects empty article content.
    with pytest.raises(
        ValueError,
        match="No readable article text",
    ):
        text_extraction.extract_from_text(empty_text)


# ============================================================
# 2. MISSING FILE
# ============================================================

def test_missing_file_is_rejected(tmp_path) -> None:
    """A missing file should raise FileNotFoundError."""

    # Create a path without creating the actual file.
    missing_file = tmp_path / "missing.txt"

    # Confirm extraction stops with the correct error.
    with pytest.raises(FileNotFoundError):
        text_extraction.extract_from_txt(missing_file)


# ============================================================
# 3. WRONG CSV COLUMN
# ============================================================

def test_missing_csv_column_is_rejected(tmp_path) -> None:
    """CSV extraction should reject an unknown column name."""

    # Create a temporary CSV file with only a headline column.
    file_path = tmp_path / "articles.csv"

    with file_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["headline"],
        )
        writer.writeheader()
        writer.writerow({"headline": "Market news"})

    # Request a column that does not exist in the CSV.
    with pytest.raises(
        ValueError,
        match="was not found",
    ):
        text_extraction.extract_from_csv(
            file_path,
            text_column="article_text",
        )


# ============================================================
# 4. EMPTY URL CONTENT
# ============================================================

def test_url_without_article_text_is_rejected(
    monkeypatch,
) -> None:
    """A webpage without paragraphs should raise a clear error."""

    class FakeResponse:
        """Controlled webpage containing no article paragraphs."""

        text = """
        <html>
            <body>
                <nav>Navigation only</nav>
            </body>
        </html>
        """

        def raise_for_status(self) -> None:
            # Simulate a successful web request.
            return None

    def fake_get(*_args, **_kwargs):
        # Return empty article HTML instead of calling the internet.
        return FakeResponse()

    # Replace requests.get so the failure is controlled and repeatable.
    monkeypatch.setattr(
        text_extraction.requests,
        "get",
        fake_get,
    )

    # Confirm a page without readable paragraphs is rejected.
    with pytest.raises(
        ValueError,
        match="No readable article text",
    ):
        text_extraction.extract_from_url(
            "https://example.com/empty"
        )
