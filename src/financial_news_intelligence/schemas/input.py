"""Schema for article data entering the analysis pipeline."""

from datetime import datetime

from pydantic import Field, HttpUrl

from .common import ProjectSchema, SourceType


class ArticleInput(ProjectSchema):
    """
    Standard article input used by all project services.

    Input:  Clean article text and source details.
    Output: One validated article object.
    Next:   Sent to company detection, market mapping, and models.
    """

    # Text extracted from pasted content, a file, or a URL.
    text: str = Field(min_length=1)

    # Record where the article came from.
    source_type: SourceType
    source_name: str | None = None
    source_url: HttpUrl | None = None

    # Keep publication time when it is available.
    published_at: datetime | None = None
