"""Bounded file extraction and address-filtered public URL reading.

Inputs and row grain
--------------------
TXT, PDF, and DOCX produce one article. CSV produces one article per accepted
row in the configured text column. URL input produces one article from public
paragraph text.

Security and cleanup
--------------------
File bytes are size-limited before parsing. URL hosts are resolved before every
request and redirect; private, loopback, link-local, multicast, reserved, and
unspecified addresses are rejected. No uploaded file is written to disk.
"""

from __future__ import annotations

import csv
import io
import ipaddress
import socket
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from .errors import ApiProblem


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".csv"}
ALLOWED_URL_SCHEMES = {"http", "https"}
ALLOWED_TEXT_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
}
MAXIMUM_PDF_PAGES = 200
MAXIMUM_DOCX_ARCHIVE_ENTRIES = 1_000
MAXIMUM_DOCX_UNCOMPRESSED_BYTES = 20 * 1024 * 1024
DOCX_DOCUMENT_PATH = "word/document.xml"
DOCX_WORD_NAMESPACE = (
    "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
)


def _load_pdf_reader_class() -> type[Any]:
    """Load the optional PDF parser only when a PDF request is received.

    PDF parsing is not needed for health, readiness, text, CSV, DOCX, or URL
    requests. Delaying this import prevents one missing optional dependency from
    stopping the whole FastAPI application during startup.
    """

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ApiProblem(
            503,
            "pdf_parser_unavailable",
            "PDF extraction failed.",
            "FastAPI PDF parser dependency",
            "The pypdf package is not installed in the active API environment.",
            "Install the pinned project dependencies in .venv, then rerun "
            "FastAPI readiness.",
        ) from exc
    return PdfReader


def _extract_docx_text(content: bytes) -> str:
    """Extract paragraph text from DOCX XML without a third-party import.

    A DOCX file is a ZIP archive. The article text is stored in
    ``word/document.xml``. Reading that XML directly keeps DOCX support
    available even when the optional ``python-docx`` package is absent. No
    archive member is written to disk.
    """

    try:
        with ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            uncompressed_bytes = sum(entry.file_size for entry in entries)
            if len(entries) > MAXIMUM_DOCX_ARCHIVE_ENTRIES:
                raise ApiProblem(
                    413,
                    "docx_entry_limit_exceeded",
                    "DOCX extraction failed.",
                    "DOCX archive entry count",
                    f"The archive contains {len(entries)} entries; the limit is "
                    f"{MAXIMUM_DOCX_ARCHIVE_ENTRIES}.",
                    "Use a simpler DOCX document and retry.",
                )
            if uncompressed_bytes > MAXIMUM_DOCX_UNCOMPRESSED_BYTES:
                raise ApiProblem(
                    413,
                    "docx_expansion_limit_exceeded",
                    "DOCX extraction failed.",
                    "DOCX uncompressed size",
                    f"The archive expands to {uncompressed_bytes} bytes; the "
                    f"limit is {MAXIMUM_DOCX_UNCOMPRESSED_BYTES}.",
                    "Remove embedded media or split the document and retry.",
                )
            try:
                document_xml = archive.read(DOCX_DOCUMENT_PATH)
            except KeyError as exc:
                raise ApiProblem(
                    422,
                    "docx_document_xml_missing",
                    "DOCX extraction failed.",
                    "DOCX document structure",
                    f"The archive does not contain {DOCX_DOCUMENT_PATH}.",
                    "Upload the original DOCX file and retry.",
                ) from exc
    except BadZipFile as exc:
        raise ApiProblem(
            422,
            "docx_archive_invalid",
            "DOCX extraction failed.",
            "DOCX archive structure",
            "The DOCX ZIP structure is invalid.",
            "Upload the original DOCX file and retry.",
        ) from exc

    try:
        root = ElementTree.fromstring(document_xml)
    except ElementTree.ParseError as exc:
        raise ApiProblem(
            422,
            "docx_xml_invalid",
            "DOCX extraction failed.",
            "DOCX document XML",
            "The main DOCX document XML is invalid.",
            "Upload the original DOCX file and retry.",
        ) from exc

    paragraph_tag = f"{{{DOCX_WORD_NAMESPACE}}}p"
    text_tag = f"{{{DOCX_WORD_NAMESPACE}}}t"
    paragraphs: list[str] = []
    for paragraph in root.iter(paragraph_tag):
        paragraph_text = "".join(
            node.text or "" for node in paragraph.iter(text_tag)
        ).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)
    return " ".join(paragraphs)


def clean_text(value: str, maximum_characters: int) -> str:
    """Normalize whitespace and enforce one explicit text limit."""

    cleaned = " ".join(str(value).strip().split())
    if not cleaned:
        raise ApiProblem(
            422,
            "empty_text",
            "Text extraction failed.",
            "Extracted article text",
            "No readable text remained after whitespace normalization.",
            "Provide a text-based document or paste readable article text.",
        )
    if len(cleaned) > maximum_characters:
        raise ApiProblem(
            413,
            "text_too_large",
            "Text validation failed.",
            "Extracted article text",
            f"The article contains {len(cleaned)} characters; the limit is "
            f"{maximum_characters}.",
            "Submit a shorter article without unrelated page content.",
        )
    return cleaned


def extract_uploaded_bytes(
    filename: str,
    content: bytes,
    maximum_bytes: int,
    maximum_characters: int,
    csv_text_column: str | None,
    maximum_csv_rows: int,
) -> list[str]:
    """Extract one or more clean texts from a supported in-memory file."""

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ApiProblem(
            415,
            "unsupported_file_type",
            "File extraction failed.",
            "Uploaded filename",
            f"The extension {extension or '[none]'} is not supported.",
            "Upload a TXT, PDF, DOCX, or CSV file.",
        )
    if not content:
        raise ApiProblem(
            422,
            "empty_file",
            "File extraction failed.",
            "Uploaded file content",
            "The uploaded file contains zero bytes.",
            "Upload a non-empty supported file.",
        )
    if len(content) > maximum_bytes:
        raise ApiProblem(
            413,
            "file_too_large",
            "File extraction failed.",
            "Uploaded file content",
            f"The file contains {len(content)} bytes; the limit is "
            f"{maximum_bytes}.",
            "Upload a smaller file or split the input into separate requests.",
        )

    try:
        if extension == ".txt":
            raw_text = content.decode("utf-8")
            return [clean_text(raw_text, maximum_characters)]
        if extension == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise ApiProblem(
                    422,
                    "pdf_signature_invalid",
                    "PDF extraction failed.",
                    "Uploaded PDF signature",
                    "The file does not begin with a valid PDF signature.",
                    "Upload the original non-encrypted PDF file.",
                )
            reader_class = _load_pdf_reader_class()
            reader = reader_class(io.BytesIO(content))
            if len(reader.pages) > MAXIMUM_PDF_PAGES:
                raise ApiProblem(
                    413,
                    "pdf_page_limit_exceeded",
                    "PDF extraction failed.",
                    "Uploaded PDF page count",
                    f"The PDF contains {len(reader.pages)} pages; the limit is "
                    f"{MAXIMUM_PDF_PAGES}.",
                    "Split the PDF into smaller documents and retry.",
                )
            raw_text = " ".join(page.extract_text() or "" for page in reader.pages)
            return [clean_text(raw_text, maximum_characters)]
        if extension == ".docx":
            if not content.startswith(b"PK"):
                raise ApiProblem(
                    422,
                    "docx_signature_invalid",
                    "DOCX extraction failed.",
                    "Uploaded DOCX signature",
                    "The file is not a valid ZIP-based DOCX document.",
                    "Upload the original DOCX file and retry.",
                )
            raw_text = _extract_docx_text(content)
            return [clean_text(raw_text, maximum_characters)]

        if not csv_text_column or not csv_text_column.strip():
            raise ApiProblem(
                422,
                "csv_column_required",
                "CSV extraction failed.",
                "csv_text_column form field",
                "A CSV file requires the name of its article-text column.",
                "Send csv_text_column with the exact CSV column name.",
            )
        decoded_csv = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(decoded_csv, newline=""))
        column = csv_text_column.strip()
        fieldnames = [str(value) for value in (reader.fieldnames or [])]
        if column not in fieldnames:
            raise ApiProblem(
                422,
                "csv_column_missing",
                "CSV extraction failed.",
                "CSV header",
                f"Column '{column}' was not found. Available columns: "
                + ", ".join(fieldnames),
                "Use an existing CSV column name and retry.",
            )
        raw_values: list[str] = []
        for row in reader:
            value = row.get(column)
            if value is not None and str(value).strip():
                raw_values.append(str(value))
        if len(raw_values) > maximum_csv_rows:
            raise ApiProblem(
                413,
                "csv_row_limit_exceeded",
                "CSV extraction failed.",
                "CSV article rows",
                f"The CSV contains {len(raw_values)} readable rows; the limit is "
                f"{maximum_csv_rows}.",
                "Split the CSV into smaller batches and retry.",
            )
        return [clean_text(value, maximum_characters) for value in raw_values]
    except ApiProblem:
        raise
    except (UnicodeDecodeError, ValueError, OSError) as exc:
        raise ApiProblem(
            422,
            "file_parse_failed",
            "File extraction failed.",
            f"{extension.upper()} parser",
            f"The file could not be parsed safely: {type(exc).__name__}.",
            "Confirm that the file is not encrypted or damaged, then retry.",
        ) from exc


def _validate_public_host(hostname: str) -> None:
    """Resolve a hostname and reject every non-public address."""

    try:
        records = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ApiProblem(
            422,
            "url_dns_failed",
            "URL extraction failed.",
            "Domain-name resolution",
            "The URL hostname could not be resolved.",
            "Check the public hostname and retry.",
        ) from exc
    addresses = {record[4][0] for record in records}
    if not addresses:
        raise ApiProblem(
            422,
            "url_dns_empty",
            "URL extraction failed.",
            "Domain-name resolution",
            "The URL hostname resolved to no addresses.",
            "Use a public HTTP or HTTPS article URL.",
        )
    for raw_address in addresses:
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError as exc:
            raise ApiProblem(
                422,
                "url_dns_address_invalid",
                "URL extraction failed.",
                "Domain-name resolution",
                "The hostname returned an invalid network address.",
                "Use a public hostname with valid IPv4 or IPv6 records.",
            ) from exc
        if any(
            (
                address.is_private,
                address.is_loopback,
                address.is_link_local,
                address.is_multicast,
                address.is_reserved,
                address.is_unspecified,
            )
        ):
            raise ApiProblem(
                403,
                "url_address_blocked",
                "URL extraction was blocked.",
                "Resolved URL address",
                "The hostname resolves to a private or non-public network address.",
                "Use a public article URL that does not redirect to an internal host.",
            )


def validate_public_url(url: str) -> str:
    """Validate URL syntax, scheme, credentials, and public host resolution."""

    parsed = urlsplit(url)
    if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
        raise ApiProblem(
            422,
            "url_scheme_invalid",
            "URL validation failed.",
            "URL scheme",
            "Only HTTP and HTTPS URLs are supported.",
            "Submit a public HTTP or HTTPS article URL.",
        )
    if parsed.username or parsed.password:
        raise ApiProblem(
            422,
            "url_credentials_forbidden",
            "URL validation failed.",
            "URL authority",
            "Embedded usernames and passwords are not allowed.",
            "Remove credentials from the URL and retry.",
        )
    if not parsed.hostname:
        raise ApiProblem(
            422,
            "url_host_missing",
            "URL validation failed.",
            "URL hostname",
            "The URL does not contain a hostname.",
            "Submit a complete public article URL.",
        )
    _validate_public_host(parsed.hostname)
    return url


def extract_public_url(
    url: str,
    timeout_seconds: int,
    maximum_bytes: int,
    maximum_characters: int,
    maximum_redirects: int,
) -> str:
    """Fetch public article text with bounded redirects and response size.

    The application validates each hostname before the request and each redirect.
    It also ignores ambient proxy and netrc settings. A later public-deployment
    phase must add network-level egress controls because application checks alone
    cannot eliminate every domain-name-system rebinding race.
    """

    current_url = validate_public_url(url)
    headers = {
        "User-Agent": "FinancialNewsIntelligence/1.0 (educational research API)"
    }
    with requests.Session() as session:
        # Ambient proxy and netrc settings can redirect traffic or attach
        # credentials. This local research API uses a direct session instead.
        session.trust_env = False
        for redirect_count in range(maximum_redirects + 1):
            try:
                response = session.get(
                    current_url,
                    timeout=timeout_seconds,
                    headers=headers,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.RequestException as exc:
                raise ApiProblem(
                    502,
                    "url_request_failed",
                    "URL extraction failed.",
                    "Remote article request",
                    f"The remote request failed: {type(exc).__name__}.",
                    "Confirm the article is publicly reachable and retry once.",
                ) from exc
            if 300 <= response.status_code < 400:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise ApiProblem(
                        502,
                        "url_redirect_missing",
                        "URL extraction failed.",
                        "Remote redirect response",
                        "The server returned a redirect without a destination.",
                        "Use the final public article URL directly.",
                    )
                if redirect_count >= maximum_redirects:
                    raise ApiProblem(
                        502,
                        "url_redirect_limit",
                        "URL extraction failed.",
                        "Remote redirect chain",
                        "The URL exceeded the configured redirect limit.",
                        "Use the final public article URL directly.",
                    )
                current_url = validate_public_url(
                    urljoin(current_url, location)
                )
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                status = response.status_code
                response.close()
                raise ApiProblem(
                    502,
                    "url_http_failed",
                    "URL extraction failed.",
                    "Remote article response",
                    f"The remote server returned HTTP {status}.",
                    "Use a publicly accessible article URL and retry once.",
                ) from exc

            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    declared_bytes = int(content_length)
                except ValueError:
                    declared_bytes = 0
                if declared_bytes > maximum_bytes:
                    response.close()
                    raise ApiProblem(
                        413,
                        "url_content_length_too_large",
                        "URL extraction failed.",
                        "Remote Content-Length header",
                        f"The server declared {declared_bytes} bytes; the limit "
                        f"is {maximum_bytes}.",
                        "Use a shorter public article page.",
                    )
            content_type = response.headers.get("Content-Type", "").split(
                ";", 1
            )[0]
            if content_type.lower() not in ALLOWED_TEXT_CONTENT_TYPES:
                response.close()
                raise ApiProblem(
                    415,
                    "url_content_type_invalid",
                    "URL extraction failed.",
                    "Remote Content-Type header",
                    f"The response type '{content_type or '[missing]'}' is not "
                    "readable text.",
                    "Use a public HTML or plain-text article URL.",
                )
            chunks: list[bytes] = []
            total_bytes = 0
            try:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total_bytes += len(chunk)
                    if total_bytes > maximum_bytes:
                        raise ApiProblem(
                            413,
                            "url_content_too_large",
                            "URL extraction failed.",
                            "Remote response body",
                            f"The response exceeded the {maximum_bytes}-byte limit.",
                            (
                                "Use a shorter article page without large embedded "
                                "content."
                            ),
                        )
                    chunks.append(chunk)
            except requests.RequestException as exc:
                raise ApiProblem(
                    502,
                    "url_stream_failed",
                    "URL extraction failed.",
                    "Remote response body",
                    f"The response stream failed: {type(exc).__name__}.",
                    "Confirm the article is publicly reachable and retry once.",
                ) from exc
            finally:
                response.close()

            encoding = response.encoding or "utf-8"
            try:
                html = b"".join(chunks).decode(encoding, errors="strict")
            except UnicodeDecodeError as exc:
                raise ApiProblem(
                    422,
                    "url_decode_failed",
                    "URL extraction failed.",
                    "Remote response decoding",
                    "The response character encoding could not be decoded safely.",
                    "Use a UTF-8 or standard text article page.",
                ) from exc
            if content_type.lower() == "text/plain":
                return clean_text(html, maximum_characters)
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(
                ["script", "style", "nav", "footer", "header", "aside"]
            ):
                tag.decompose()
            paragraph_text = " ".join(
                paragraph.get_text(" ", strip=True)
                for paragraph in soup.find_all("p")
                if paragraph.get_text(strip=True)
            )
            return clean_text(paragraph_text, maximum_characters)

    raise ApiProblem(
        502,
        "url_redirect_state_invalid",
        "URL extraction failed.",
        "Redirect processor",
        "The redirect loop ended without a readable response.",
        "Use the final public article URL directly.",
    )
