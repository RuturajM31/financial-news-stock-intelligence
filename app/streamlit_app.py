"""Launch the four-page public Financial News Sentiment Analyzer.

Streamlit Community Cloud and local development both start here. The public
renderer owns page configuration, navigation, lazy Full BERT loading, and the
four supported pages.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep the entry point intentionally small: only the retained public app loads.
from app.public_cloud_app import render_public_streamlit_cloud_app


def main() -> None:
    """Render the supported public Streamlit application."""

    render_public_streamlit_cloud_app(PROJECT_ROOT)


if __name__ == "__main__":
    main()
