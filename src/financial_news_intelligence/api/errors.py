"""Structured API errors that state failure, location, cause, and next step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi.responses import JSONResponse


@dataclass
class ApiProblem(RuntimeError):
    """One safe client-facing failure description."""

    status_code: int
    error_code: str
    what_failed: str
    where_failed: str
    why_failed: str
    safe_next_step: str

    def __str__(self) -> str:
        return self.why_failed

    def payload(self, request_id: str | None = None) -> dict[str, Any]:
        """Return the stable public error schema without a stack trace."""

        return {
            "status": "FAILED",
            "error_code": self.error_code,
            "what_failed": self.what_failed,
            "where_failed": self.where_failed,
            "why_failed": self.why_failed,
            "safe_next_step": self.safe_next_step,
            "request_id": request_id,
        }


def problem_response(
    problem: ApiProblem,
    request_id: str | None,
) -> JSONResponse:
    """Convert an ApiProblem into a JSON response."""

    return JSONResponse(
        status_code=problem.status_code,
        content=problem.payload(request_id),
    )
