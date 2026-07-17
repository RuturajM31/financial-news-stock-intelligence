"""Dependency-free parsing for bounded FastAPI file-upload requests.

Purpose
-------
FastAPI normally requires the optional ``python-multipart`` package when a
route uses ``UploadFile``, ``File``, or ``Form`` parameters. The project's
current main environment does not contain that package. This module therefore
reads the request body directly and parses the small multipart envelope with
the Python standard library.

Inputs and downstream use
-------------------------
The input is one Starlette request. Standard browser and API-client multipart
uploads use a form field named ``file`` and may include ``csv_text_column``.
Raw-body clients may instead send ``X-Filename`` and
``X-CSV-Text-Column`` headers. The returned bytes are passed to
``extract_uploaded_bytes``; they are never written to disk.

Limits and failure behavior
---------------------------
The body, part count, form-field size, filename length, and file size are all
bounded. Unknown or duplicate fields fail closed. Every controlled failure
states what failed, where it failed, why it failed, and the exact safe next
step through ``ApiProblem``.
"""

from __future__ import annotations

from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as default_email_policy
from pathlib import Path

from starlette.requests import Request

from .errors import ApiProblem


MULTIPART_CONTENT_TYPE = "multipart/form-data"
FILE_FIELD_NAME = "file"
CSV_TEXT_COLUMN_FIELD_NAME = "csv_text_column"
RAW_FILENAME_HEADER = "X-Filename"
RAW_CSV_COLUMN_HEADER = "X-CSV-Text-Column"
MAXIMUM_MULTIPART_OVERHEAD_BYTES = 256 * 1024
MAXIMUM_MULTIPART_PARTS = 4
MAXIMUM_FORM_FIELD_BYTES = 1_024
MAXIMUM_FILENAME_CHARACTERS = 255


@dataclass(frozen=True)
class ParsedUploadRequest:
    """One validated in-memory file request and its optional CSV column."""

    filename: str
    content: bytes
    csv_text_column: str | None


def _request_size_problem(
    actual_bytes: int,
    maximum_bytes: int,
) -> ApiProblem:
    """Build one consistent request-size failure."""

    return ApiProblem(
        413,
        "upload_request_too_large",
        "File upload failed.",
        "FastAPI file-upload request body",
        f"The request contains {actual_bytes} bytes; the safe limit is "
        f"{maximum_bytes} bytes.",
        "Upload a smaller file without unrelated form fields and retry.",
    )


async def _read_bounded_body(
    request: Request,
    maximum_bytes: int,
) -> bytes:
    """Read one request body without accepting bytes beyond the named limit."""

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_bytes = int(content_length)
        except ValueError as exc:
            raise ApiProblem(
                400,
                "content_length_invalid",
                "File upload failed.",
                "HTTP Content-Length header",
                "The Content-Length header is not an integer.",
                "Send the request again with a valid Content-Length header.",
            ) from exc
        if declared_bytes < 0:
            raise ApiProblem(
                400,
                "content_length_invalid",
                "File upload failed.",
                "HTTP Content-Length header",
                "The Content-Length header cannot be negative.",
                "Send the request again with a valid Content-Length header.",
            )
        if declared_bytes > maximum_bytes:
            raise _request_size_problem(declared_bytes, maximum_bytes)

    chunks: list[bytes] = []
    accumulated_bytes = 0
    async for chunk in request.stream():
        if not chunk:
            continue
        accumulated_bytes += len(chunk)
        if accumulated_bytes > maximum_bytes:
            raise _request_size_problem(accumulated_bytes, maximum_bytes)
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_filename(filename: str) -> str:
    """Reject blank, path-like, or unreasonably long upload filenames."""

    safe_filename = filename.strip()
    if not safe_filename:
        raise ApiProblem(
            422,
            "upload_filename_missing",
            "File upload failed.",
            "Uploaded file metadata",
            "The upload does not contain a filename.",
            "Choose a TXT, PDF, DOCX, or CSV file and submit it again.",
        )
    if len(safe_filename) > MAXIMUM_FILENAME_CHARACTERS:
        raise ApiProblem(
            422,
            "upload_filename_too_long",
            "File upload failed.",
            "Uploaded file metadata",
            f"The filename contains {len(safe_filename)} characters; the limit is "
            f"{MAXIMUM_FILENAME_CHARACTERS}.",
            "Rename the file with a shorter plain filename and retry.",
        )
    if "\x00" in safe_filename or "/" in safe_filename or "\\" in safe_filename:
        raise ApiProblem(
            422,
            "upload_filename_invalid",
            "File upload failed.",
            "Uploaded file metadata",
            "The filename contains a path separator or a null character.",
            "Upload the file with a plain filename such as article.txt.",
        )
    if Path(safe_filename).name != safe_filename:
        raise ApiProblem(
            422,
            "upload_filename_invalid",
            "File upload failed.",
            "Uploaded file metadata",
            "The filename is not a plain basename.",
            "Upload the file with a plain filename such as article.txt.",
        )
    return safe_filename


def _decode_form_field(field_name: str, payload: bytes, charset: str) -> str:
    """Decode one bounded text form field with strict character handling."""

    if len(payload) > MAXIMUM_FORM_FIELD_BYTES:
        raise ApiProblem(
            413,
            "upload_form_field_too_large",
            "File upload failed.",
            f"Multipart form field {field_name}",
            f"The field contains {len(payload)} bytes; the limit is "
            f"{MAXIMUM_FORM_FIELD_BYTES}.",
            "Send a short CSV column name and retry.",
        )
    try:
        return payload.decode(charset, errors="strict").strip()
    except (LookupError, UnicodeDecodeError) as exc:
        raise ApiProblem(
            422,
            "upload_form_field_invalid",
            "File upload failed.",
            f"Multipart form field {field_name}",
            f"The field is not valid text using the declared {charset} charset.",
            "Send the field as UTF-8 text and retry.",
        ) from exc


def _parse_multipart_body(
    content_type: str,
    body: bytes,
    maximum_file_bytes: int,
) -> ParsedUploadRequest:
    """Parse one bounded multipart body without ``python-multipart``."""

    if "\r" in content_type or "\n" in content_type:
        raise ApiProblem(
            400,
            "multipart_content_type_invalid",
            "File upload failed.",
            "HTTP Content-Type header",
            "The multipart Content-Type header contains a line break.",
            "Send a standard multipart/form-data request and retry.",
        )

    synthetic_message = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n\r\n"
    ).encode("latin-1") + body
    parsed_message = BytesParser(policy=default_email_policy).parsebytes(
        synthetic_message
    )
    if not parsed_message.is_multipart():
        raise ApiProblem(
            400,
            "multipart_body_invalid",
            "File upload failed.",
            "Multipart request structure",
            "The request body does not match its multipart Content-Type header.",
            "Send one standard multipart file field named file and retry.",
        )

    multipart_parts = list(parsed_message.iter_parts())
    if len(multipart_parts) > MAXIMUM_MULTIPART_PARTS:
        raise ApiProblem(
            413,
            "multipart_part_limit_exceeded",
            "File upload failed.",
            "Multipart request part count",
            f"The request contains {len(multipart_parts)} parts; the limit is "
            f"{MAXIMUM_MULTIPART_PARTS}.",
            "Send one file and, for CSV only, one csv_text_column field.",
        )

    filename: str | None = None
    file_content: bytes | None = None
    csv_text_column: str | None = None
    seen_names: set[str] = set()

    for part in multipart_parts:
        if part.get_content_disposition() != "form-data":
            raise ApiProblem(
                422,
                "multipart_disposition_invalid",
                "File upload failed.",
                "Multipart request part",
                "Every request part must use form-data content disposition.",
                "Send one standard multipart file field named file and retry.",
            )
        raw_name = part.get_param("name", header="content-disposition")
        field_name = str(raw_name or "").strip()
        if field_name not in {FILE_FIELD_NAME, CSV_TEXT_COLUMN_FIELD_NAME}:
            raise ApiProblem(
                422,
                "multipart_field_unknown",
                "File upload failed.",
                "Multipart request field",
                f"The field {field_name or '<missing>'} is not supported.",
                "Use file and, for CSV only, csv_text_column.",
            )
        if field_name in seen_names:
            raise ApiProblem(
                422,
                "multipart_field_duplicate",
                "File upload failed.",
                f"Multipart request field {field_name}",
                "The field appears more than once.",
                "Send each accepted field exactly once.",
            )
        seen_names.add(field_name)

        decoded_payload = part.get_payload(decode=True)
        payload = decoded_payload if isinstance(decoded_payload, bytes) else b""
        part_filename = part.get_filename()
        if field_name == FILE_FIELD_NAME:
            if part_filename is None:
                raise ApiProblem(
                    422,
                    "upload_filename_missing",
                    "File upload failed.",
                    "Multipart file field",
                    "The file field does not contain a filename parameter.",
                    "Choose a TXT, PDF, DOCX, or CSV file and retry.",
                )
            filename = _validate_filename(str(part_filename))
            if len(payload) > maximum_file_bytes:
                raise _request_size_problem(len(payload), maximum_file_bytes)
            file_content = payload
        else:
            if part_filename is not None:
                raise ApiProblem(
                    422,
                    "csv_column_field_invalid",
                    "File upload failed.",
                    "csv_text_column multipart field",
                    "The CSV column selector must be a text field, not a file.",
                    "Send csv_text_column as short UTF-8 text and retry.",
                )
            charset = part.get_content_charset() or "utf-8"
            decoded_column = _decode_form_field(field_name, payload, charset)
            csv_text_column = decoded_column or None

    if filename is None or file_content is None:
        raise ApiProblem(
            422,
            "upload_file_field_missing",
            "File upload failed.",
            "Multipart request fields",
            "The required file field is missing.",
            "Send one multipart file field named file and retry.",
        )
    return ParsedUploadRequest(
        filename=filename,
        content=file_content,
        csv_text_column=csv_text_column,
    )


async def read_upload_request(
    request: Request,
    maximum_file_bytes: int,
) -> ParsedUploadRequest:
    """Read a multipart or raw-body upload without optional parser packages."""

    content_type = request.headers.get("content-type", "").strip()
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type == MULTIPART_CONTENT_TYPE:
        maximum_request_bytes = (
            maximum_file_bytes + MAXIMUM_MULTIPART_OVERHEAD_BYTES
        )
        request_body = await _read_bounded_body(request, maximum_request_bytes)
        return _parse_multipart_body(
            content_type,
            request_body,
            maximum_file_bytes,
        )

    raw_filename = request.headers.get(RAW_FILENAME_HEADER, "")
    filename = _validate_filename(raw_filename)
    request_body = await _read_bounded_body(request, maximum_file_bytes)
    csv_column_header = request.headers.get(RAW_CSV_COLUMN_HEADER)
    csv_text_column = csv_column_header.strip() if csv_column_header else None
    if (
        csv_text_column
        and len(csv_text_column.encode("utf-8")) > MAXIMUM_FORM_FIELD_BYTES
    ):
        raise ApiProblem(
            413,
            "upload_form_field_too_large",
            "File upload failed.",
            f"HTTP {RAW_CSV_COLUMN_HEADER} header",
            "The CSV column name exceeds the safe field-size limit.",
            "Send a shorter CSV column name and retry.",
        )
    return ParsedUploadRequest(
        filename=filename,
        content=request_body,
        csv_text_column=csv_text_column,
    )
