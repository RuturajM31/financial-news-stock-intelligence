"""Define and validate settings used by the Streamlit presentation layer.

The settings in this module describe the browser page and the location of the
approved CSS file. They do not include secrets, model paths, provider values,
or FastAPI credentials. Package 2 will add a separate, private API setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_ALLOWED_LAYOUTS = frozenset({"centered", "wide"})
_ALLOWED_SIDEBAR_STATES = frozenset({"auto", "expanded", "collapsed"})


@dataclass(frozen=True)
class AppSettings:
    """Store stable presentation settings for one Streamlit application.

    Attributes:
        project_root: Absolute project directory used to locate local assets.
        page_title: Browser-tab title shown to the user.
        page_icon: Small browser-tab symbol. No external image is required.
        layout: Streamlit page width mode.
        initial_sidebar_state: Sidebar state on the first page load.
        css_path: Approved local stylesheet loaded by ``layout.py``.
        expected_streamlit_version: Exact version recorded in the requirement
            file. Runtime verification belongs to the later environment gate.
    """

    project_root: Path
    page_title: str
    page_icon: str
    layout: str
    initial_sidebar_state: str
    css_path: Path
    expected_streamlit_version: str

    def validate(self) -> None:
        """Raise a clear error when a setting is missing or unsafe.

        The CSS path must remain inside the project root. This prevents a
        changed setting from reading an unrelated local file into the page.
        """

        if not self.page_title.strip():
            raise ValueError("The Streamlit page title must not be empty.")
        if self.layout not in _ALLOWED_LAYOUTS:
            raise ValueError(f"Unsupported Streamlit layout: {self.layout!r}.")
        if self.initial_sidebar_state not in _ALLOWED_SIDEBAR_STATES:
            raise ValueError(
                "Unsupported initial sidebar state: "
                f"{self.initial_sidebar_state!r}."
            )

        resolved_root = self.project_root.resolve()
        resolved_css = self.css_path.resolve()
        if not resolved_css.is_relative_to(resolved_root):
            raise ValueError("The premium stylesheet must stay inside the project.")
        if not resolved_css.is_file():
            raise FileNotFoundError("The premium Streamlit stylesheet is missing.")


def get_app_settings(project_root: Path | None = None) -> AppSettings:
    """Return validated, deterministic settings for the Streamlit app.

    Args:
        project_root: Project directory. When omitted, it is derived from this
            module so tests and local runs use the same location rule.

    Returns:
        A validated immutable ``AppSettings`` value.
    """

    resolved_root = (
        project_root.resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    settings = AppSettings(
        project_root=resolved_root,
        page_title="Financial News Intelligence | Ruturaj Mokashi",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
        css_path=resolved_root / "app" / "styles" / "premium_theme.css",
        expected_streamlit_version="1.58.0",
    )
    settings.validate()
    return settings
