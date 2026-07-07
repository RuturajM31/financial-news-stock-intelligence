"""Prepare leakage-safe article splits for Transformer training."""

from collections import defaultdict
from datetime import timezone
from typing import Sequence

from financial_news_intelligence.schemas.training_data import (
    DatasetSplit,
)
from financial_news_intelligence.schemas.transformer_data import (
    TransformerExample,
    TransformerSplitConfig,
)


# ============================================================
# 1. REMOVE IDENTICAL ARTICLES
# ============================================================

def remove_exact_duplicates(
    examples: Sequence[TransformerExample],
) -> tuple[TransformerExample, ...]:
    """
    Remove articles that contain exactly the same normalized text.

    Input:
        Article cards.

    Output:
        Article cards without exact text duplicates.

    Why:
        The model should not see the same article more than once.
    """

    unique_examples: list[TransformerExample] = []
    seen_texts: set[str] = set()

    # Sort first so duplicate removal is reproducible.
    ordered_examples = sorted(
        examples,
        key=lambda example: (
            example.published_at.astimezone(timezone.utc),
            example.article_id,
        ),
    )

    for example in ordered_examples:
        # Ignore differences in capitals and repeated spaces.
        normalized_text = " ".join(
            example.text.lower().split()
        )

        # Skip this article when the same text was already stored.
        if normalized_text in seen_texts:
            continue

        seen_texts.add(normalized_text)
        unique_examples.append(example)

    return tuple(unique_examples)


# ============================================================
# 2. GROUP ARTICLES BY REAL-WORLD EVENT
# ============================================================

def group_examples_by_event(
    examples: Sequence[TransformerExample],
) -> dict[str, tuple[TransformerExample, ...]]:
    """
    Put articles about the same event into one group.

    Example:
        NVIDIA press release
        Reuters NVIDIA article
        Another NVIDIA earnings article

    All three may share:
        event_id = nvidia_q2_earnings
    """

    event_groups: dict[
        str,
        list[TransformerExample],
    ] = defaultdict(list)

    for example in examples:
        event_groups[example.event_id].append(example)

    return {
        event_id: tuple(
            sorted(
                event_examples,
                key=lambda example: (
                    example.published_at.astimezone(
                        timezone.utc
                    ),
                    example.article_id,
                ),
            )
        )
        for event_id, event_examples in event_groups.items()
    }


# ============================================================
# 3. CALCULATE HOW MANY EVENTS GO INTO EACH SPLIT
# ============================================================

def calculate_event_split_counts(
    event_count: int,
    config: TransformerSplitConfig,
) -> tuple[int, int, int]:
    """
    Calculate train, validation, and test event counts.

    At least one event must remain in every split.
    """

    if event_count < 3:
        raise ValueError(
            "At least three unique events are required."
        )

    train_count = max(
        1,
        int(event_count * config.train_ratio),
    )

    validation_count = max(
        1,
        int(event_count * config.validation_ratio),
    )

    # Keep reducing the larger split until one test event remains.
    while train_count + validation_count >= event_count:
        if train_count > 1:
            train_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            raise ValueError(
                "Not enough events for three dataset splits."
            )

    test_count = (
        event_count
        - train_count
        - validation_count
    )

    return train_count, validation_count, test_count


# ============================================================
# 4. MAIN CHRONOLOGICAL SPLIT FUNCTION
# ============================================================

def split_transformer_examples(
    examples: Sequence[TransformerExample],
    config: TransformerSplitConfig | None = None,
) -> dict[DatasetSplit, tuple[TransformerExample, ...]]:
    """
    Divide article cards into chronological event-level splits.

    Rules:
    1. Remove exact duplicate articles.
    2. Keep every event inside one split.
    3. Oldest events go to training.
    4. Middle events go to validation.
    5. Newest events go to testing.
    """

    if config is None:
        config = TransformerSplitConfig()

    if not examples:
        raise ValueError(
            "At least one Transformer example is required."
        )

    unique_examples = remove_exact_duplicates(
        examples
    )

    event_groups = group_examples_by_event(
        unique_examples
    )

    # Sort events using the first publication time in each group.
    ordered_event_ids = sorted(
        event_groups,
        key=lambda event_id: (
            event_groups[event_id][0]
            .published_at
            .astimezone(timezone.utc),
            event_id,
        ),
    )

    train_count, validation_count, _ = (
        calculate_event_split_counts(
            len(ordered_event_ids),
            config,
        )
    )

    train_end = train_count
    validation_end = train_count + validation_count

    split_event_ids = {
        DatasetSplit.TRAIN: ordered_event_ids[
            :train_end
        ],
        DatasetSplit.VALIDATION: ordered_event_ids[
            train_end:validation_end
        ],
        DatasetSplit.TEST: ordered_event_ids[
            validation_end:
        ],
    }

    final_splits: dict[
        DatasetSplit,
        tuple[TransformerExample, ...],
    ] = {}

    for split, event_ids in split_event_ids.items():
        split_examples: list[TransformerExample] = []

        for event_id in event_ids:
            for example in event_groups[event_id]:
                # Create a copy and attach its assigned split.
                split_examples.append(
                    example.model_copy(
                        update={"split": split}
                    )
                )

        final_splits[split] = tuple(
            split_examples
        )

    return final_splits
