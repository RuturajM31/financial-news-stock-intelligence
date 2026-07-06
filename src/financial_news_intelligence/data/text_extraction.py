"""Extract readable financial-news text from supported input sources."""

from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader


# ============================================================
# 1. SHARED VALIDATION AND CLEANING
# ============================================================

def validate_file(file_path: Path) -> None:
    """Confirm that an uploaded file exists before reading it."""

    # Stop early with a clear message when the file is missing.
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Reject folders because extraction requires a real file.
    if not file_path.is_file():
        raise ValueError(f"Expected a file, but received: {file_path}")


def clean_extracted_text(text: str) -> str:
    """
    Remove unnecessary whitespace from extracted article text.

    Input:  Raw text from pasted content, files, or a webpage.
    Output: Clean text with single spaces.
    Next:   Sent to company detection and prediction models.
    """

    # Replace repeated spaces, tabs, and line breaks with one space.
    cleaned_text = " ".join(text.strip().split())

    if not cleaned_text:
        raise ValueError("No readable article text was found.")

    return cleaned_text


# ============================================================
# 2. PASTED TEXT
# ============================================================

def extract_from_text(text: str) -> str:
    """Clean article text pasted directly by the user."""

    return clean_extracted_text(text)


# ============================================================
# 3. TXT FILES
# ============================================================

def extract_from_txt(file_path: Path) -> str:
    """Read and clean article text from a TXT file."""

    validate_file(file_path)

    # UTF-8 is the standard encoding used by the project.
    raw_text = file_path.read_text(encoding="utf-8")

    return clean_extracted_text(raw_text)


# ============================================================
# 4. PDF FILES
# ============================================================

def extract_from_pdf(file_path: Path) -> str:
    """Extract and clean text from every readable PDF page."""

    validate_file(file_path)
    reader = PdfReader(file_path)

    # Some PDF pages may contain images and return no text.
    page_text = [
        page.extract_text() or ""
        for page in reader.pages
    ]

    return clean_extracted_text(" ".join(page_text))


# ============================================================
# 5. DOCX FILES
# ============================================================

def extract_from_docx(file_path: Path) -> str:
    """Extract and clean paragraph text from a DOCX file."""

    validate_file(file_path)
    document = Document(file_path)

    # Ignore empty paragraphs before joining the document text.
    paragraph_text = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return clean_extracted_text(" ".join(paragraph_text))


# ============================================================
# 6. CSV FILES
# ============================================================

def extract_from_csv(
    file_path: Path,
    text_column: str,
) -> list[str]:
    """
    Extract multiple articles from one CSV text column.

    Input:  CSV file path and the name of its article-text column.
    Output: One clean article string per valid row.
    Next:   Used for batch prediction or dataset preparation.
    """

    validate_file(file_path)
    dataframe = pd.read_csv(file_path)

    # Stop when the requested article column does not exist.
    if text_column not in dataframe.columns:
        available_columns = ", ".join(dataframe.columns)

        raise ValueError(
            f"Column '{text_column}' was not found. "
            f"Available columns: {available_columns}"
        )

    # Remove missing rows and clean every remaining article.
    articles = [
        clean_extracted_text(str(value))
        for value in dataframe[text_column].dropna()
        if str(value).strip()
    ]

    if not articles:
        raise ValueError(
            f"Column '{text_column}' contains no readable articles."
        )

    return articles


# ============================================================
# 7. FINANCIAL-NEWS URLS
# ============================================================

def extract_from_url(
    url: str,
    timeout_seconds: int = 20,
) -> str:
    """Download a webpage and extract readable paragraph text."""

    # Identify the request clearly and prevent endless waiting.
    response = requests.get(
        url,
        timeout=timeout_seconds,
        headers={
            "User-Agent": (
                "FinancialNewsIntelligence/0.1 "
                "(educational project)"
            )
        },
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove page elements that are not part of the article.
    for unwanted_tag in soup(
        ["script", "style", "nav", "footer", "header", "aside"]
    ):
        unwanted_tag.decompose()

    # Financial-news articles are usually stored inside paragraph tags.
    paragraph_text = [
        paragraph.get_text(" ", strip=True)
        for paragraph in soup.find_all("p")
        if paragraph.get_text(strip=True)
    ]

    return clean_extracted_text(" ".join(paragraph_text))
