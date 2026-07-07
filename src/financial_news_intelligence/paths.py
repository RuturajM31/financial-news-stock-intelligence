"""Central project paths used across the application."""

from pathlib import Path


# Move upward from this file to find the project root folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Store source and prepared datasets in separate folders.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REFERENCE_DATA_DIR = DATA_DIR / "reference"

# Store trained models and reusable machine-learning files.
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
TOKENIZERS_DIR = ARTIFACTS_DIR / "tokenizers"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"

# Store charts, tables, predictions, and final reports.
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
INTELLIGENCE_REPORTS_DIR = REPORTS_DIR / "intelligence"


def ensure_project_directories() -> None:
    """
    Create all required runtime folders.

    Output: Missing folders are created.
    Next:   Project files can be saved without path errors.
    """

    # Keep every folder in one list so setup stays easy to maintain.
    required_directories = (
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        REFERENCE_DATA_DIR,
        MODELS_DIR,
        TOKENIZERS_DIR,
        EMBEDDINGS_DIR,
        MANIFESTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        PREDICTIONS_DIR,
        INTELLIGENCE_REPORTS_DIR,
    )

    # Create missing folders and leave existing folders unchanged.
    for directory in required_directories:
        directory.mkdir(parents=True, exist_ok=True)
