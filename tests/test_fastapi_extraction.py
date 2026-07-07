"""Tests for bounded file and URL input security."""

import io
import socket
import sys
from zipfile import ZipFile

import pytest

from financial_news_intelligence.api.errors import ApiProblem
from financial_news_intelligence.api.extraction import (
    extract_uploaded_bytes,
    validate_public_url,
)


def test_txt_extraction_cleans_readable_text() -> None:
    """Prepare TXT bytes, run extraction, and check normalized output."""

    result = extract_uploaded_bytes(
        filename="article.txt",
        content=b"Company   results\n improved.",
        maximum_bytes=1024,
        maximum_characters=100,
        csv_text_column=None,
        maximum_csv_rows=5,
    )

    assert result == ["Company results improved."]


def test_unsupported_file_extension_fails_closed() -> None:
    """Prepare an executable extension, run extraction, and check rejection."""

    with pytest.raises(ApiProblem) as captured:
        extract_uploaded_bytes(
            filename="article.exe",
            content=b"not allowed",
            maximum_bytes=1024,
            maximum_characters=100,
            csv_text_column=None,
            maximum_csv_rows=5,
        )

    assert captured.value.error_code == "unsupported_file_type"


def test_private_url_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prepare loopback DNS, run URL validation, and check SSRF prevention."""

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))
        ],
    )

    with pytest.raises(ApiProblem) as captured:
        validate_public_url("http://example.test/article")

    assert captured.value.error_code == "url_address_blocked"


def test_csv_extraction_returns_one_text_per_accepted_row() -> None:
    """Prepare two CSV rows, run extraction, and check stable row grain."""

    result = extract_uploaded_bytes(
        filename="articles.csv",
        content=b"headline\nRevenue increased.\nCosts declined.\n",
        maximum_bytes=1024,
        maximum_characters=100,
        csv_text_column="headline",
        maximum_csv_rows=5,
    )

    assert result == ["Revenue increased.", "Costs declined."]


def test_docx_extraction_reads_paragraph_text_without_python_docx() -> None:
    """Prepare DOCX XML, run extraction, and check dependency-free output."""

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body><w:p><w:r><w:t>Revenue increased during the quarter.</w:t>'
        '</w:r></w:p></w:body></w:document>'
    )
    buffer = io.BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        archive.writestr("word/document.xml", document_xml)

    result = extract_uploaded_bytes(
        filename="article.docx",
        content=buffer.getvalue(),
        maximum_bytes=1024 * 1024,
        maximum_characters=100,
        csv_text_column=None,
        maximum_csv_rows=5,
    )

    assert result == ["Revenue increased during the quarter."]


def test_pdf_extraction_reads_page_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prepare a PDF signature, run extraction, and check bounded page text."""

    class FakePage:
        """Return deterministic text without creating a binary test fixture."""

        def extract_text(self) -> str:
            return "Revenue increased during the quarter."

    class FakeReader:
        """Expose one page through the same interface as pypdf.PdfReader."""

        def __init__(self, _stream: object) -> None:
            self.pages = [FakePage()]

    monkeypatch.setattr(
        "financial_news_intelligence.api.extraction._load_pdf_reader_class",
        lambda: FakeReader,
    )

    result = extract_uploaded_bytes(
        filename="article.pdf",
        content=b"%PDF-1.7\n",
        maximum_bytes=1024,
        maximum_characters=100,
        csv_text_column=None,
        maximum_csv_rows=5,
    )

    assert result == ["Revenue increased during the quarter."]


def test_pdf_missing_dependency_returns_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepare a missing parser, run PDF extraction, and check safe guidance."""

    monkeypatch.setitem(sys.modules, "pypdf", None)

    with pytest.raises(ApiProblem) as captured:
        extract_uploaded_bytes(
            filename="article.pdf",
            content=b"%PDF-1.7\n",
            maximum_bytes=1024,
            maximum_characters=100,
            csv_text_column=None,
            maximum_csv_rows=5,
        )

    assert captured.value.error_code == "pdf_parser_unavailable"
    assert (
        "Install the pinned project dependencies"
        in captured.value.safe_next_step
    )
