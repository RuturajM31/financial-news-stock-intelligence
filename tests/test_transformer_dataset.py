"""Tests for leakage-safe Transformer dataset splitting."""

from datetime import datetime, timezone

import pytest

from financial_news_intelligence.data.transformer_dataset import (
    calculate_event_split_counts,
    remove_exact_duplicates,
    split_transformer_examples,
)
from financial_news_intelligence.schemas.common import (
    SentimentLabel,
)
from financial_news_intelligence.schemas.training_data import (
    DatasetSplit,
)
from financial_news_intelligence.schemas.transformer_data import (
    TransformerExample,
    TransformerSplitConfig,
)


# ============================================================
# 1. TEST DATA HELPER
# ============================================================

def make_example(
    article_number: int,
    event_number: int,
    month: int,
    text: str | None = None,
) -> TransformerExample:
    """
    Create one small article card for testing.

    Prepare:
        Article ID, event ID, text, label, and publication date.

    Output:
        One valid TransformerExample.
    """

    if text is None:
        text = (
            f"Company {event_number} reported financial results "
            f"for test article {article_number}."
        )

    return TransformerExample(
        article_id=f"article_{article_number:03d}",
        event_id=f"event_{event_number:03d}",
        text=text,
        label=SentimentLabel.BULLISH,
        published_at=datetime(
            2024,
            month,
            1,
            9,
            0,
            tzinfo=timezone.utc,
        ),
        ticker="TEST",
        source_id="verified_test_source",
    )


# ============================================================
# 2. EXACT DUPLICATE REMOVAL
# ============================================================

def test_exact_duplicate_articles_are_removed() -> None:
    """The same normalized article text should appear only once."""

    # Prepare two articles containing the same words.
    examples = [
        make_example(
            article_number=1,
            event_number=1,
            month=1,
            text="Company A reported very strong quarterly revenue.",
        ),
        make_example(
            article_number=2,
            event_number=2,
            month=2,
            text="  COMPANY A reported very strong quarterly revenue.  ",
        ),
    ]

    # Run duplicate removal.
    result = remove_exact_duplicates(examples)

    # Check that only the earliest article remains.
    assert len(result) == 1
    assert result[0].article_id == "article_001"


# ============================================================
# 3. SAME EVENT MUST STAY TOGETHER
# ============================================================

def test_articles_from_same_event_stay_in_one_split() -> None:
    """One event must never appear in multiple dataset splits."""

    # Prepare six events.
    # Event 1 contains two different articles.
    examples = [
        make_example(1, 1, 1),
        make_example(
            2,
            1,
            1,
            text=(
                "A second publisher reported the same company "
                "earnings event."
            ),
        ),
        make_example(3, 2, 2),
        make_example(4, 3, 3),
        make_example(5, 4, 4),
        make_example(6, 5, 5),
        make_example(7, 6, 6),
    ]

    # Run chronological event-level splitting.
    splits = split_transformer_examples(examples)

    # Find every split containing event_001.
    event_locations = [
        split
        for split, records in splits.items()
        if any(
            record.event_id == "event_001"
            for record in records
        )
    ]

    # Both event_001 articles must remain in exactly one split.
    assert event_locations == [DatasetSplit.TRAIN]

    train_event_001_records = [
        record
        for record in splits[DatasetSplit.TRAIN]
        if record.event_id == "event_001"
    ]

    assert len(train_event_001_records) == 2


# ============================================================
# 4. CHRONOLOGICAL ORDER
# ============================================================

def test_old_events_train_and_new_events_test() -> None:
    """Older events should train the model; newest events test it."""

    # Prepare six events from January through June.
    examples = [
        make_example(1, 1, 1),
        make_example(2, 2, 2),
        make_example(3, 3, 3),
        make_example(4, 4, 4),
        make_example(5, 5, 5),
        make_example(6, 6, 6),
    ]

    # Run the split.
    splits = split_transformer_examples(examples)

    # Check the expected chronological event placement.
    train_events = {
        record.event_id
        for record in splits[DatasetSplit.TRAIN]
    }

    validation_events = {
        record.event_id
        for record in splits[DatasetSplit.VALIDATION]
    }

    test_events = {
        record.event_id
        for record in splits[DatasetSplit.TEST]
    }

    assert train_events == {
        "event_001",
        "event_002",
        "event_003",
        "event_004",
    }

    assert validation_events == {"event_005"}
    assert test_events == {"event_006"}


# ============================================================
# 5. SPLIT NAME IS ATTACHED
# ============================================================

def test_each_article_receives_its_split_name() -> None:
    """Every returned article should know its assigned split."""

    # Prepare the minimum three unique events.
    examples = [
        make_example(1, 1, 1),
        make_example(2, 2, 2),
        make_example(3, 3, 3),
    ]

    # Run the split.
    splits = split_transformer_examples(examples)

    # Check that every article carries the correct split value.
    for split, records in splits.items():
        for record in records:
            assert record.split == split


# ============================================================
# 6. TOO FEW EVENTS
# ============================================================

def test_fewer_than_three_events_are_rejected() -> None:
    """Train, validation, and test each need at least one event."""

    config = TransformerSplitConfig()

    with pytest.raises(
        ValueError,
        match="At least three unique events",
    ):
        calculate_event_split_counts(
            event_count=2,
            config=config,
        )


# ============================================================
# 7. EMPTY INPUT
# ============================================================

def test_empty_article_collection_is_rejected() -> None:
    """The splitting function cannot work without articles."""

    with pytest.raises(
        ValueError,
        match="At least one Transformer example",
    ):
        split_transformer_examples([])
