"""Public schemas used across the project."""

# Shared labels and validation rules.
from .common import (
    MarketSession,
    MovementLabel,
    SentimentLabel,
    SourceType,
)

# Data entering the system.
from .input import ArticleInput

# Information discovered by the system.
from .context import (
    ArticleContext,
    CompanyContext,
    FinancialContext,
    TradingContext,
)

# Results returned by the system.
from .output import (
    ForecastResult,
    HistoricalReturnRange,
    ImportantPhrase,
    IntelligenceResult,
    ModelReference,
    MovementProbabilities,
    SentimentProbabilities,
    SentimentResult,
    SimilarArticle,
)

# Control which schemas are available through this package.
__all__ = [
    "ArticleContext",
    "ArticleInput",
    "CompanyContext",
    "FinancialContext",
    "ForecastResult",
    "HistoricalReturnRange",
    "ImportantPhrase",
    "IntelligenceResult",
    "MarketSession",
    "ModelReference",
    "MovementLabel",
    "MovementProbabilities",
    "SentimentLabel",
    "SentimentProbabilities",
    "SentimentResult",
    "SimilarArticle",
    "SourceType",
    "TradingContext",
]

# Public provenance models.
from .provenance import (
    DataPurpose,
    SourceAssessment,
    SourceProvenance,
    VerificationStatus,
)

# Public market-data models.
from .market_data import (
    MarketPriceHistory,
    PriceBar,
    PriceCrossCheck,
    ReturnLabel,
)

# Public verified training-dataset models.
from .training_data import (
    DatasetManifest,
    DatasetSplit,
    NewsReactionRecord,
)

# Public historical-reaction intelligence models.
from .historical_intelligence import ReactionCohort

# Public investment-scenario models.
from .investment import (
    InvestmentOutcome,
    InvestmentRequest,
    InvestmentScenarioResult,
    ScenarioLevel,
)


# Public Transformer dataset models.
from .transformer_data import (
    TransformerExample,
    TransformerSplitConfig,
)
