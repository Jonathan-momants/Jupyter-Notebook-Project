"""Process a Momants CSV export into starting and ending sentiment per conversation.

Sentiment here is the visitor's attitude *towards the service*, not the severity of
their situation. A calmly phrased report of a lost phone, a broken wristband or a
failed payment is Neutral: the visitor is doing a task, not complaining. Only
dissatisfaction with the help itself -- repetition, escalation, anger markers,
"this does not answer my question" -- counts as Negative (frustrated).

A general-purpose review-trained sentiment model cannot make that distinction: it
scores topical polarity ("lost", "broken", "did not work") and therefore labels most
ordinary service questions Negative. This module uses a SetFit classifier trained on
Momants' own labelled examples instead, the same approach as momants_intent.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import shutil
from typing import Iterable

from datasets import Dataset
import numpy as np
import pandas as pd
from setfit import SetFitModel, Trainer, TrainingArguments

from momants_conversation_filter import (
    BARE_URL,
    select_visitor_messages,
)
from momants_privacy import mask_pii


BASIS_MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TRAINING_DATA_PATH = Path("data/training/sentiment_training.csv")
LOCAL_MODEL_PATH = Path("model/momants-sentiment")

# Neutral is the resting state of a service conversation, so it doubles as the
# "no confident signal" answer. Anything the model is unsure about lands here
# rather than being reported as a mood the visitor never expressed.
NEUTRAL_LABEL = "Neutral (task-oriented)"
SENTIMENT_CATEGORIES = [
    NEUTRAL_LABEL,
    "Negative (frustrated)",
    "Positive",
]
# Neutral is the default answer, so the two signal classes each carry their own
# decision threshold rather than competing on argmax. This is deliberate: the
# classes are rare (roughly 3% frustrated in real traffic), so an argmax over a
# Neutral-dominated distribution almost never fires. Reading each signal class
# against its own bar makes the operating point explicit and tunable without
# retraining -- lower the frustration bar to catch more escalations at the cost
# of more false alerts, raise it for a quieter dashboard.
# Tuned on the held-out Claude-labelled set (validate_momants_sentiment.py). Both
# sit inside a flat region of their sweep rather than on a knife edge: frustration
# is stable from 0.50 to 0.90, enthusiasm from 0.20 to 0.35. Re-check them after
# any retrain -- SetFit training is stochastic, and with only 4 frustrated
# examples in the held-out set these estimates carry real noise.
DECISION_THRESHOLDS = {
    "Negative (frustrated)": 0.50,
    "Positive": 0.30,
}
DECISION_ORDER = ["Negative (frustrated)", "Positive"]

SAFE_COLUMNS = [
    "created_at",
    "text",
    "from_agent",
    "message_type",
    "conversation_id",
    "agent_id",
]

OUTPUT_COLUMNS = [
    "conversation_id",
    "customer_message_count",
    "starting_sentiment",
    "starting_confidence",
    "ending_sentiment",
    "ending_confidence",
    "has_trend",
    "explanation",
]

REQUIRED_COLUMNS = {
    "created_at",
    "text",
    "from_agent",
    "conversation_id",
}

# Positions of safe fields only in the headerless 22-field format.
HEADERLESS_POSITIONS = {
    0: "created_at",
    3: "text",
    8: "from_agent",
    10: "message_type",
    17: "agent_id",
    19: "conversation_id",
}

COLUMN_ALIASES = {
    "createdat": "created_at",
    "created": "created_at",
    "timestamp": "created_at",
    "text": "text",
    "message": "text",
    "message_text": "text",
    "fromagent": "from_agent",
    "is_agent": "from_agent",
    "messagetype": "message_type",
    "type": "message_type",
    "conversationid": "conversation_id",
    "conversation": "conversation_id",
    "agentid": "agent_id",
}

# A closing "ja graag", "ok bedankt" or "thanks" carries no mood of its own, but it
# is very often the last visitor message in a conversation. Reading the ending
# sentiment off it replaces the visitor's actual final state with a politeness
# formula, so the ending sentiment is taken from the last message that still has
# content. If a conversation consists only of these, the last message is used after
# all and the model -- trained on the same phrases -- returns Neutral.
ACKNOWLEDGEMENT_WORDS = frozenset(
    {
        # confirmation / refusal
        "ja", "jaa", "jaaa", "nee", "neen", "yes", "yep", "no", "nope",
        "ok", "oke", "okee", "okay", "prima", "klopt", "duidelijk",
        "goed", "helder", "akkoord", "alright", "sure", "correct",
        # politeness / greetings / closings that carry no appraisal
        "graag", "alsjeblieft", "alstublieft", "please", "bitte",
        "hallo", "hoi", "hey", "hi", "goedemorgen", "goedemiddag",
        "goedenavond", "morgen", "middag", "avond", "dag", "dagje",
        "doei", "bye", "ciao", "groetjes", "tot", "ziens", "later", "verder",
        # filler that only ever appears inside the above
        "je", "u", "jou", "wel", "hoor", "dan", "dus", "das", "dat", "die",
        "het", "is", "was", "ik", "we", "me", "mij", "nog", "al", "alles",
        "that", "all", "it", "so", "much", "very", "a", "lot", "for", "the",
        "en", "and", "of", "maar", "even", "zeker", "weet", "meer", "niks",
    }
)
# Gratitude ("thanks", "bedankt", "dankjewel", "merci") and appraisal ("top",
# "fijne dag") are deliberately NOT in this set. Momants' own answer key scores
# them Positive, so they carry a real signal and must reach the classifier rather
# than being skipped as filler. Only bare confirmations, refusals and greetings
# are treated as contentless.
# A message counts as a bare acknowledgement only when *every* word in it is one
# of the above and it stays short. "Ja" and "Nee dankjewel" qualify; "Ja linkje
# mag zeker" and "Nee dat klopt niet, ik heb al drie keer gebeld" do not, because
# they carry words with real content.
MAX_ACKNOWLEDGEMENT_WORDS = 6
WORD_PATTERN = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
DIACRITICS = str.maketrans(
    {"ä": "a", "ö": "o", "ü": "u", "é": "e", "è": "e", "ê": "e", "ç": "c", "ß": "s"}
)


def _normalize_column_name(name: object) -> str:
    """Normalize a column name for comparison with known names."""
    text = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    compact = text.replace("_", "")
    return COLUMN_ALIASES.get(text, COLUMN_ALIASES.get(compact, text))


def _has_momants_header(csv_path: Path) -> bool:
    """Check whether the first CSV row contains recognizable Momants column names."""
    columns = pd.read_csv(csv_path, nrows=0).columns
    normalized = {_normalize_column_name(column) for column in columns}
    return REQUIRED_COLUMNS.issubset(normalized)


def _read_with_header(csv_path: Path) -> pd.DataFrame:
    """Read only safe fields from a CSV with column headers."""
    original_columns = pd.read_csv(csv_path, nrows=0).columns
    column_mapping = {
        column: _normalize_column_name(column)
        for column in original_columns
        if _normalize_column_name(column) in SAFE_COLUMNS
    }
    dataframe = pd.read_csv(
        csv_path,
        usecols=list(column_mapping),
        on_bad_lines="skip",
    )
    return dataframe.rename(columns=column_mapping)


def _read_without_header(csv_path: Path) -> pd.DataFrame:
    """Read only safe column positions from the 22-field format."""
    positions = list(HEADERLESS_POSITIONS)
    dataframe = pd.read_csv(
        csv_path,
        header=None,
        usecols=positions,
        on_bad_lines="skip",
        engine="python",
    )
    return dataframe.rename(columns=HEADERLESS_POSITIONS)


def load_momants_csv(source: str | Path) -> pd.DataFrame:
    """Load a local Momants export and retain only safe columns."""
    if isinstance(source, str) and re.match(r"^https?://", source, re.IGNORECASE):
        raise ValueError(
            "Endpoint URLs are not yet supported; provide a local CSV path."
        )

    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    dataframe = (
        _read_with_header(path)
        if _has_momants_header(path)
        else _read_without_header(path)
    )

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "The CSV is missing required Momants fields: "
            f"{', '.join(sorted(missing_columns))}."
        )

    for optional_column in {"message_type", "agent_id"} - set(dataframe.columns):
        dataframe[optional_column] = pd.NA

    dataframe = dataframe[SAFE_COLUMNS].copy()
    masked_message_count = 0
    masking_count = 0
    masked_texts: list[object] = []
    for value in dataframe["text"]:
        if pd.isna(value):
            masked_texts.append(value)
            continue
        masked_text, count = mask_pii(str(value))
        masked_texts.append(masked_text)
        masked_message_count += int(count > 0)
        masking_count += count
    dataframe["text"] = masked_texts
    print(f"Messages with masked PII: {masked_message_count}")
    print(f"PII values masked: {masking_count}")

    dataframe["created_at"] = pd.to_datetime(dataframe["created_at"], errors="coerce", utc=True)
    dataframe["from_agent"] = (
        dataframe["from_agent"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    # Keep unreadable from_agent values visible so each analysis can report them.
    dataframe = dataframe.dropna(subset=["created_at", "conversation_id"])
    dataframe["conversation_id"] = dataframe["conversation_id"].astype(str).str.strip()
    dataframe = dataframe.loc[dataframe["conversation_id"].ne("")].copy()

    return dataframe.sort_values(["conversation_id", "created_at"]).reset_index(drop=True)


def select_customer_messages(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep chronologically sorted, usable customer messages."""
    return select_visitor_messages(dataframe)


def _validate_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the training schema and the three permitted sentiment labels."""
    required_columns = {"text", "sentiment"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(
            f"The training CSV is missing columns: {', '.join(sorted(missing))}."
        )

    clean = data[["text", "sentiment"]].copy()
    if clean.isna().any().any():
        raise ValueError("The training data contains missing text or sentiment values.")
    clean["text"] = clean["text"].astype(str).str.strip()
    clean["sentiment"] = clean["sentiment"].astype(str).str.strip()
    if clean["text"].eq("").any() or clean["sentiment"].eq("").any():
        raise ValueError("The training data contains empty text or sentiment values.")

    unknown = set(clean["sentiment"]) - set(SENTIMENT_CATEGORIES)
    if unknown:
        raise ValueError(
            f"Unknown sentiment labels in the training data: {', '.join(sorted(unknown))}."
        )

    absent = set(SENTIMENT_CATEGORIES) - set(clean["sentiment"])
    if absent:
        raise ValueError(
            f"Training examples are missing for: {', '.join(sorted(absent))}."
        )
    return clean


def train_model(
    training_data_path: str | Path = TRAINING_DATA_PATH,
    model_path: str | Path = LOCAL_MODEL_PATH,
) -> Path:
    """Train SetFit on Momants' own labelled examples and save the model locally."""
    source = Path(training_data_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Training file not found: {source}")

    training = _validate_training_data(pd.read_csv(source, keep_default_na=False))
    label_to_index = {
        label: index for index, label in enumerate(SENTIMENT_CATEGORIES)
    }
    dataset = Dataset.from_dict(
        {
            "text": training["text"].tolist(),
            "label": training["sentiment"].map(label_to_index).tolist(),
        }
    )

    model = SetFitModel.from_pretrained(
        BASIS_MODEL_ID,
        labels=SENTIMENT_CATEGORIES,
    )
    arguments = TrainingArguments(
        output_dir="model/checkpoints-sentiment",
        batch_size=16,
        num_epochs=(1, 16),
        num_iterations=2,
        report_to="none",
        save_strategy="no",
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=dataset)
    trainer.train()

    target = Path(model_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}-new")
    if staging.exists():
        shutil.rmtree(staging)

    try:
        model.save_pretrained(staging)
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    counts = (
        training["sentiment"]
        .value_counts()
        .reindex(SENTIMENT_CATEGORIES, fill_value=0)
    )
    print(f"Training examples used: {len(training)}")
    print("Examples per category:")
    for sentiment, count in counts.items():
        print(f"- {sentiment}: {int(count)}")
    return target


def _model_labels(model: SetFitModel) -> list[str]:
    labels = [str(label) for label in model.labels]
    if set(labels) != set(SENTIMENT_CATEGORIES):
        raise ValueError(
            "The local model does not contain exactly the three required sentiment labels."
        )
    return labels


def _decide(
    probabilities: "np.ndarray",
    labels: list[str],
    thresholds: dict[str, float],
) -> tuple[list[str], list[float]]:
    """Pick the first signal class that clears its own bar, else Neutral."""
    label_positions = {label: index for index, label in enumerate(labels)}
    decisions: list[str] = []
    confidences: list[float] = []
    for row in probabilities:
        chosen = NEUTRAL_LABEL
        for label in DECISION_ORDER:
            score = float(row[label_positions[label]])
            if score >= thresholds.get(label, 1.0):
                chosen = label
                break
        decisions.append(chosen)
        confidences.append(round(float(row[label_positions[chosen]]), 4))
    return decisions, confidences


def classify_messages(
    customer_messages: pd.DataFrame,
    model_path: str | Path = LOCAL_MODEL_PATH,
    batch_size: int = 32,
    model: SetFitModel | None = None,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Classify each customer message, returning Neutral when no signal class fires.

    ``message_confidence`` is the model's probability for the label that was
    actually assigned, so a Neutral row reports how neutral the message looked --
    not how confident the strongest rejected class was.
    """
    if customer_messages.empty:
        result = customer_messages.copy()
        result["message_sentiment"] = pd.Series(dtype="object")
        result["message_confidence"] = pd.Series(dtype="float64")
        return result

    if model is None:
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(
                f"Trained sentiment model not found in {path}. "
                "Run first: python momants_sentiment.py --train"
            )
        model = SetFitModel.from_pretrained(path, local_files_only=True)

    labels = _model_labels(model)
    probabilities = model.predict_proba(
        customer_messages["text"].astype(str).tolist(),
        batch_size=batch_size,
        as_numpy=True,
        show_progress_bar=True,
    )
    decisions, confidences = _decide(
        probabilities,
        labels,
        thresholds if thresholds is not None else DECISION_THRESHOLDS,
    )

    result = customer_messages.copy()
    result["message_sentiment"] = decisions
    result["message_confidence"] = confidences
    return result


def _is_backchannel(text: object) -> bool:
    """Return whether a message is a bare acknowledgement with no mood of its own."""
    if pd.isna(text):
        return True
    words = WORD_PATTERN.findall(str(text).strip().casefold().translate(DIACRITICS))
    if not words:
        # Emoji-only or punctuation-only messages carry no classifiable text.
        return True
    if len(words) > MAX_ACKNOWLEDGEMENT_WORDS:
        return False
    return all(word in ACKNOWLEDGEMENT_WORDS for word in words)


def _select_boundary_messages(
    customer_messages: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Select the first message and the last message that still carries content.

    Returns the rows to classify plus, per conversation_id, the row label of the
    opening message and of the closing message. Both are Series keyed by
    conversation_id so every later lookup aligns by conversation rather than by
    position.
    """
    sorted_messages = customer_messages.sort_values(
        ["conversation_id", "created_at"], kind="stable"
    )
    groups = sorted_messages.groupby("conversation_id", sort=False)
    first_row = groups.head(1)
    first_indices = pd.Series(
        first_row.index, index=first_row["conversation_id"], name="first_index"
    )

    with_content = sorted_messages.loc[~sorted_messages["text"].map(_is_backchannel)]
    closing_row = with_content.groupby("conversation_id", sort=False).tail(1)
    last_indices = pd.Series(
        closing_row.index, index=closing_row["conversation_id"], name="last_index"
    )
    # Conversations that are nothing but acknowledgements keep their real last
    # message; the classifier reports Neutral for it.
    fallback_row = groups.tail(1)
    fallback_indices = pd.Series(
        fallback_row.index, index=fallback_row["conversation_id"], name="last_index"
    )
    last_indices = last_indices.reindex(first_indices.index)
    last_indices = last_indices.fillna(fallback_indices.reindex(first_indices.index))
    last_indices = last_indices.astype(first_indices.dtype)

    boundary_indices = pd.Index(
        list(dict.fromkeys([*first_indices.tolist(), *last_indices.tolist()]))
    )
    return sorted_messages.loc[boundary_indices].copy(), first_indices, last_indices


def create_conversation_summary(
    customer_messages: pd.DataFrame,
    batch_size: int = 32,
    model_path: str | Path = LOCAL_MODEL_PATH,
    model: SetFitModel | None = None,
) -> pd.DataFrame:
    """Determine sentiment for the first and last meaningful customer message."""
    if customer_messages.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    boundary_messages, first_indices, last_indices = _select_boundary_messages(
        customer_messages
    )
    classified = classify_messages(
        boundary_messages,
        model_path=model_path,
        batch_size=batch_size,
        model=model,
    )

    counts = (
        customer_messages.groupby("conversation_id", sort=False)
        .size()
        .rename("customer_message_count")
    )
    summary = counts.to_frame()
    for column, indices in (("starting", first_indices), ("ending", last_indices)):
        picked = classified.loc[indices.values]
        picked.index = indices.index
        summary[f"{column}_sentiment"] = picked["message_sentiment"]
        summary[f"{column}_confidence"] = picked["message_confidence"]
    # A conversation whose start and end are the same message has no trend to
    # report. Dashboards that chart sentiment movement must exclude these rather
    # than count them as "unchanged".
    summary["has_trend"] = (
        first_indices.ne(last_indices).reindex(summary.index).fillna(False)
    )
    summary = summary.reset_index()
    summary["explanation"] = summary.apply(
        lambda row: (
            f"The conversation starts {row['starting_sentiment'].lower()} and ends "
            f"{row['ending_sentiment'].lower()}, based on "
            f"{row['customer_message_count']} "
            f"{'usable customer message' if row['customer_message_count'] == 1 else 'usable customer messages'}."
            + ("" if row["has_trend"] else " Start and end are the same message, so there is no trend.")
        ),
        axis=1,
    )
    return summary[OUTPUT_COLUMNS]


def process_csv(
    csv_path: str | Path,
    output_directory: str | Path = "results",
    batch_size: int = 32,
    model_path: str | Path = LOCAL_MODEL_PATH,
) -> pd.DataFrame:
    """Run processing and write one safe conversation table."""
    started_at = datetime.now().astimezone()
    dataframe = load_momants_csv(csv_path)
    customer_messages = select_customer_messages(dataframe)
    conversation_summary = create_conversation_summary(
        customer_messages,
        batch_size=batch_size,
        model_path=model_path,
    )

    output_path = Path(output_directory).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
    result_path = output_path / f"sentiment_per_conversation_{timestamp}.csv"
    conversation_summary.to_csv(result_path, index=False)
    conversation_summary.attrs["output_path"] = result_path.resolve()

    return conversation_summary


# Backward-compatible programmatic aliases for the original public API.
laad_momants_csv = load_momants_csv
selecteer_bezoekersberichten = select_customer_messages
classificeer_berichten = classify_messages
maak_gespreksoverzicht = create_conversation_summary


def verwerk_csv(
    csv_pad: str | Path,
    uitvoermap: str | Path = "results",
    batchgrootte: int = 32,
) -> pd.DataFrame:
    """Backward-compatible wrapper for :func:`process_csv`."""
    return process_csv(csv_pad, uitvoermap, batchgrootte)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract sentiment from a Momants CSV export.",
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        help="Path to the Momants CSV export.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model once on the configured training CSV.",
    )
    parser.add_argument(
        "--training-data",
        type=Path,
        default=TRAINING_DATA_PATH,
        help="CSV with the text and sentiment columns.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=LOCAL_MODEL_PATH,
        help="Directory containing the trained model.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results"),
        help="Directory for sentiment_per_conversation_<timestamp>.csv (default: results).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of messages per model batch (default: 32).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check loading and grouping without starting the model.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _create_parser().parse_args(argv)

    if arguments.train:
        model_path = train_model(
            training_data_path=arguments.training_data,
            model_path=arguments.model_path,
        )
        print(f"Sentiment model saved to: {model_path.resolve()}")
        return 0

    if arguments.csv_path is None:
        raise SystemExit("Provide a CSV path, or use --train.")

    if arguments.check_only:
        dataframe = load_momants_csv(arguments.csv_path)
        customer_messages = select_customer_messages(dataframe)
        print(f"Message rows loaded: {len(dataframe)}")
        print(f"Usable customer messages: {len(customer_messages)}")
        print(f"Conversations: {customer_messages['conversation_id'].nunique()}")
        return 0

    conversations = process_csv(
        csv_path=arguments.csv_path,
        output_directory=arguments.output_directory,
        batch_size=arguments.batch_size,
        model_path=arguments.model_path,
    )
    print(f"Conversation results: {len(conversations)}")
    print(f"Output written to: {conversations.attrs['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
