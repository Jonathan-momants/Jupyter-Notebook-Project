"""Train and use a SetFit classifier for Momants conversation intents."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import shutil
from typing import Iterable

from datasets import Dataset
import pandas as pd
from setfit import SetFitModel, Trainer, TrainingArguments

from momants_conversation_filter import (
    select_visitor_messages as select_filtered_visitor_messages,
)
from momants_sentiment import load_momants_csv


BASIS_MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TRAINING_DATA_PATH = Path("data/training/intent_training.csv")
LOCAL_MODEL_PATH = Path("model/momants-intentie")

INTENT_CATEGORIES = [
    "Request Information",
    "Resolve Problem or Incident",
    "Perform Transaction / Change",
    "Active Navigation Help",
    "Operate System",
    "Report Emergency",
    "None",
]

OUTPUT_COLUMNS = [
    "conversation_id",
    "intent",
    "confidence",
    "first_detected_at",
]

BARE_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)


def _valideer_trainingsdata(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the training schema and the six permitted intent labels."""
    vereiste_kolommen = {"text", "intent"}
    ontbrekend = vereiste_kolommen - set(data.columns)
    if ontbrekend:
        raise ValueError(
            "The training CSV is missing columns: "
            f"{', '.join(sorted(ontbrekend))}."
        )

    schoon = data[["text", "intent"]].copy()
    if schoon.isna().any().any():
        raise ValueError(
            "The training data contains missing text or intent values."
        )
    schoon["text"] = schoon["text"].astype(str).str.strip()
    schoon["intent"] = schoon["intent"].astype(str).str.strip()
    if schoon["text"].eq("").any() or schoon["intent"].eq("").any():
        raise ValueError(
            "The training data contains empty text or intent values."
        )

    onbekend = set(schoon["intent"]) - set(INTENT_CATEGORIES)
    if onbekend:
        raise ValueError(
            "Unknown intent labels in the training data: "
            f"{', '.join(sorted(onbekend))}."
        )

    ontbrekende_labels = set(INTENT_CATEGORIES) - set(schoon["intent"])
    if ontbrekende_labels:
        raise ValueError(
            "Training examples are missing for: "
            f"{', '.join(sorted(ontbrekende_labels))}."
        )
    return schoon


def train_model(
    training_data_path: str | Path = TRAINING_DATA_PATH,
    model_path: str | Path = LOCAL_MODEL_PATH,
) -> Path:
    """Train SetFit on handwritten examples and save the model locally."""
    bron = Path(training_data_path).expanduser()
    if not bron.is_file():
        raise FileNotFoundError(f"Training file not found: {bron}")

    training = _valideer_trainingsdata(
        pd.read_csv(bron, keep_default_na=False)
    )
    label_naar_index = {
        label: index for index, label in enumerate(INTENT_CATEGORIES)
    }
    dataset = Dataset.from_dict(
        {
            "text": training["text"].tolist(),
            "label": training["intent"].map(label_naar_index).tolist(),
        }
    )

    model = SetFitModel.from_pretrained(
        BASIS_MODEL_ID,
        labels=INTENT_CATEGORIES,
    )
    argumenten = TrainingArguments(
        output_dir="model/checkpoints-intentie",
        batch_size=16,
        num_epochs=(1, 16),
        num_iterations=2,
        report_to="none",
        save_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=argumenten,
        train_dataset=dataset,
    )
    trainer.train()

    doel = Path(model_path).expanduser()
    doel.parent.mkdir(parents=True, exist_ok=True)
    tijdelijk_doel = doel.with_name(f".{doel.name}-nieuw")
    if tijdelijk_doel.exists():
        shutil.rmtree(tijdelijk_doel)

    try:
        model.save_pretrained(tijdelijk_doel)
        if doel.exists():
            shutil.rmtree(doel)
        tijdelijk_doel.replace(doel)
    except Exception:
        if tijdelijk_doel.exists():
            shutil.rmtree(tijdelijk_doel)
        raise

    aantallen = (
        training["intent"]
        .value_counts()
        .reindex(INTENT_CATEGORIES, fill_value=0)
    )
    print(f"Training examples used: {len(training)}")
    print("Examples per category:")
    for intentie, aantal in aantallen.items():
        print(f"- {intentie}: {int(aantal)}")
    return doel


def selecteer_bezoekersberichten(data: pd.DataFrame) -> pd.DataFrame:
    """Select only usable visitor messages in chronological order."""
    return select_filtered_visitor_messages(data)


select_visitor_messages = selecteer_bezoekersberichten


def _modellabels(model: SetFitModel) -> list[str]:
    labels = [str(label) for label in model.labels]
    if set(labels) != set(INTENT_CATEGORIES):
        raise ValueError(
            "The local model does not contain exactly the six required intent labels."
        )
    return labels


def classificeer_berichten(
    bezoekersberichten: pd.DataFrame,
    model_pad: str | Path = LOCAL_MODEL_PATH,
    batchgrootte: int = 32,
    model: SetFitModel | None = None,
) -> pd.DataFrame:
    """Classify each usable visitor message separately."""
    if bezoekersberichten.empty:
        leeg = bezoekersberichten.copy()
        leeg["intent"] = pd.Series(dtype="object")
        leeg["confidence"] = pd.Series(dtype="float64")
        return leeg

    if model is None:
        pad = Path(model_pad).expanduser()
        if not pad.is_dir():
            raise FileNotFoundError(
                f"Trained intent model not found in {pad}. "
                "Run first: python momants_intent.py --train"
            )
        model = SetFitModel.from_pretrained(pad, local_files_only=True)

    labels = _modellabels(model)
    kansen = model.predict_proba(
        bezoekersberichten["text"].tolist(),
        batch_size=batchgrootte,
        as_numpy=True,
        show_progress_bar=True,
    )
    beste_indices = kansen.argmax(axis=1)
    beste_scores = kansen.max(axis=1)

    resultaat = bezoekersberichten.copy()
    resultaat["intent"] = [labels[index] for index in beste_indices]
    resultaat["confidence"] = beste_scores
    return resultaat


classify_messages = classificeer_berichten


def maak_intentieoverzicht(
    geclassificeerd: pd.DataFrame,
) -> pd.DataFrame:
    """Keep the first detection per unique conversation-intent combination."""
    if geclassificeerd.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    echte_intenties = geclassificeerd.loc[
        geclassificeerd["intent"].ne("None")
    ].copy()
    if echte_intenties.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    eerste = (
        echte_intenties.sort_values(
            ["conversation_id", "created_at"], kind="stable"
        )
        .drop_duplicates(["conversation_id", "intent"], keep="first")
        .rename(columns={"created_at": "first_detected_at"})
    )
    eerste["first_detected_at"] = eerste[
        "first_detected_at"
    ].apply(lambda waarde: waarde.isoformat())
    eerste["confidence"] = eerste["confidence"].round(4)
    return eerste[OUTPUT_COLUMNS].reset_index(drop=True)


create_intent_summary = maak_intentieoverzicht


def process_csv(
    csv_path: str | Path,
    output_directory: str | Path = "results",
    model_path: str | Path = LOCAL_MODEL_PATH,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Classify a Momants export and write the conversation intents."""
    gestart_op = datetime.now().astimezone()
    data = load_momants_csv(csv_path)
    bezoekersberichten = selecteer_bezoekersberichten(data)
    geclassificeerd = classificeer_berichten(
        bezoekersberichten,
        model_pad=model_path,
        batchgrootte=batch_size,
    )
    overzicht = maak_intentieoverzicht(geclassificeerd)

    doelmap = Path(output_directory).expanduser()
    doelmap.mkdir(parents=True, exist_ok=True)
    tijdstempel = gestart_op.strftime("%Y%m%d_%H%M%S_%f")
    uitvoerpad = doelmap / f"intents_per_conversation_{tijdstempel}.csv"
    overzicht.to_csv(uitvoerpad, index=False)
    overzicht.attrs["output_path"] = uitvoerpad.resolve()
    return overzicht


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or use the Momants intent classifier."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        help="Path to the local Momants CSV export.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model once on the configured training CSV.",
    )
    parser.add_argument(
        "--training-data",
        dest="training_data",
        type=Path,
        default=TRAINING_DATA_PATH,
        help="CSV with the text and intent columns.",
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        type=Path,
        default=LOCAL_MODEL_PATH,
        help="Directory containing the trained model.",
    )
    parser.add_argument(
        "--output-directory",
        dest="output_directory",
        type=Path,
        default=Path("results"),
        help="Directory for intents_per_conversation_<timestamp>.csv.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=32,
        help="Number of visitor messages per model batch.",
    )
    parser.add_argument(
        "--check-only",
        dest="check_only",
        action="store_true",
        help="Only validate loading, without loading the model.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.train:
        model_pad = train_model(
            training_data_path=args.training_data,
            model_path=args.model_path,
        )
        print(f"Intent model saved to: {model_pad.resolve()}")
        return 0

    if args.csv_path is None:
        raise SystemExit("Provide a CSV path, or use --train.")

    if args.check_only:
        data = load_momants_csv(args.csv_path)
        bezoekers = selecteer_bezoekersberichten(data)
        print(f"Message rows read: {len(data)}")
        print(f"Usable visitor messages: {len(bezoekers)}")
        print(f"Conversations: {bezoekers['conversation_id'].nunique()}")
        return 0

    overzicht = process_csv(
        csv_path=args.csv_path,
        output_directory=args.output_directory,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )
    print(f"Conversation intents: {len(overzicht)}")
    print(f"Output written to: {overzicht.attrs['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())