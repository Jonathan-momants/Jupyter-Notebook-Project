"""Repeatable validation for the Momants embedding-based topic classifier."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import unicodedata
from typing import Iterable

import pandas as pd

import momants_topic


NONE_LABEL = "None"


def normaliseer_tekst(value: object) -> str:
    """Lowercase text, remove punctuation, and collapse whitespace."""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in text
    )
    return re.sub(r"\s+", " ", text).strip()


def _normaliseer_kolomnaam(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _vind_kolom(
    kolommen: list[str],
    voorkeursnamen: list[str],
    *,
    bevat: tuple[str, ...] = (),
) -> str:
    genormaliseerd = {
        _normaliseer_kolomnaam(kolom): kolom for kolom in kolommen
    }
    for naam in voorkeursnamen:
        if naam in genormaliseerd:
            return genormaliseerd[naam]
    kandidaten = [
        origineel
        for normaal, origineel in genormaliseerd.items()
        if all(deel in normaal for deel in bevat)
    ]
    if len(kandidaten) == 1:
        return kandidaten[0]
    raise ValueError(
        "Could not uniquely identify an answer-key column. "
        f"Available columns: {', '.join(kolommen)}."
    )


def laad_antwoordsleutel(pad: str | Path) -> pd.DataFrame:
    """Read the answer key and repair a surplus unquoted comma in its text field."""
    bron = Path(pad).expanduser()
    if not bron.is_file():
        raise FileNotFoundError(f"Answer key not found: {bron}")

    with bron.open("r", encoding="utf-8-sig", newline="") as bestand:
        rijen = list(csv.reader(bestand))
    if not rijen:
        raise ValueError("The answer key is empty.")

    kop = [waarde.strip() for waarde in rijen[0]]
    text_kolom = _vind_kolom(
        kop,
        ["text", "message", "bericht", "testzin", "sentence"],
        bevat=("text",),
    )
    text_index = kop.index(text_kolom)
    gerepareerd: list[list[str]] = []
    for regelnummer, rij in enumerate(rijen[1:], start=2):
        if not rij or not any(waarde.strip() for waarde in rij):
            continue
        if len(rij) > len(kop):
            extra = len(rij) - len(kop)
            rij = (
                rij[:text_index]
                + [",".join(rij[text_index : text_index + extra + 1])]
                + rij[text_index + extra + 1 :]
            )
        if len(rij) != len(kop):
            raise ValueError(
                f"Answer-key row {regelnummer} has {len(rij)} fields; "
                f"expected {len(kop)}."
            )
        gerepareerd.append(rij)

    sleutel = pd.DataFrame(gerepareerd, columns=kop)
    main_kolom = _vind_kolom(
        kop,
        [
            "new_main_topic",
            "main_topic_new",
            "main_topic",
            "nieuw_hoofdonderwerp",
            "hoofdonderwerp_nieuw",
            "hoofdonderwerp",
        ],
        bevat=("main", "topic"),
    )
    sub_kolom = _vind_kolom(
        kop,
        ["subtopic", "sub_topic", "subonderwerp"],
        bevat=("sub",),
    )

    sleutel = sleutel[[text_kolom, main_kolom, sub_kolom]].rename(
        columns={
            text_kolom: "answer_text",
            main_kolom: "true_main_topic",
            sub_kolom: "true_subtopic",
        }
    )
    sleutel["normalized_text"] = sleutel["answer_text"].map(normaliseer_tekst)
    sleutel["true_main_topic"] = (
        sleutel["true_main_topic"].astype(str).str.strip()
    )
    sleutel["true_subtopic"] = sleutel["true_subtopic"].astype(str).str.strip()
    sleutel.loc[
        sleutel["true_main_topic"].str.lower().eq("none"), "true_main_topic"
    ] = NONE_LABEL
    sleutel.loc[
        sleutel["true_subtopic"].str.lower().eq("none"), "true_subtopic"
    ] = NONE_LABEL

    if sleutel["normalized_text"].eq("").any():
        raise ValueError("The answer key contains empty text.")

    labelparen = sleutel.groupby("normalized_text", sort=False)[
        ["true_main_topic", "true_subtopic"]
    ].nunique()
    conflicten = labelparen.loc[
        labelparen["true_main_topic"].gt(1) | labelparen["true_subtopic"].gt(1)
    ]
    if not conflicten.empty:
        raise ValueError(
            "Identical normalized answer-key texts have conflicting labels: "
            + ", ".join(conflicten.index.tolist())
        )
    return sleutel.drop_duplicates("normalized_text", keep="first")


def _percentage(masker: pd.Series) -> float:
    return float(masker.mean() * 100.0)


def valideer(
    csv_pad: str | Path,
    antwoordsleutel_pad: str | Path,
    seed_pad: str | Path = momants_topic.TOPICS_SEED_PATH,
    event_id: str = momants_topic.EVENT_ID,
    batchgrootte: int = 16,
) -> dict[str, float]:
    """Classify the test export and print metrics, confusion matrix, and errors."""
    data = momants_topic.laad_momants_csv(csv_pad)
    berichten = momants_topic.selecteer_bezoekersberichten(data)
    onderwerpen = momants_topic.laad_onderwerpen(seed_pad, event_id)
    voorspeld = momants_topic.classificeer_berichten(
        berichten,
        onderwerpen,
        batchgrootte=batchgrootte,
    )
    voorspeld["normalized_text"] = voorspeld["text"].map(normaliseer_tekst)

    sleutel = laad_antwoordsleutel(antwoordsleutel_pad)
    onbekende_sleutelteksten = set(sleutel["normalized_text"]) - set(
        voorspeld["normalized_text"]
    )
    if onbekende_sleutelteksten:
        print(
            f"Warning: {len(onbekende_sleutelteksten)} answer-key texts do not "
            "occur in this test export and will not be scored."
        )

    evaluatie = voorspeld.merge(
        sleutel[
            ["normalized_text", "true_main_topic", "true_subtopic"]
        ],
        on="normalized_text",
        how="left",
        validate="many_to_one",
    )
    zonder_antwoord = evaluatie["true_main_topic"].isna()
    if zonder_antwoord.any():
        voorbeelden = evaluatie.loc[zonder_antwoord, "text"].head(5).tolist()
        raise ValueError(
            f"{int(zonder_antwoord.sum())} visitor messages have no answer-key "
            f"match. Examples: {voorbeelden}"
        )

    toegestane_waarheid = set(momants_topic.MAIN_TOPICS) | {NONE_LABEL}
    onbekende_labels = set(evaluatie["true_main_topic"]) - toegestane_waarheid
    if onbekende_labels:
        raise ValueError(
            "Unknown true main-topic labels: "
            f"{', '.join(sorted(onbekende_labels))}."
        )

    inhoudelijk = evaluatie["true_main_topic"].ne(NONE_LABEL)
    smalltalk = ~inhoudelijk
    onderwerp_eval = evaluatie.loc[inhoudelijk].copy()
    if onderwerp_eval.empty:
        raise ValueError("The answer key contains no topical messages.")
    if not smalltalk.any():
        raise ValueError("The answer key contains no None/smalltalk messages.")

    main_correct = onderwerp_eval["main_topic"].eq(
        onderwerp_eval["true_main_topic"]
    )
    hoofd_accuracy = _percentage(main_correct)

    recalls: list[float] = []
    for hoofdonderwerp in momants_topic.MAIN_TOPICS:
        waar_masker = onderwerp_eval["true_main_topic"].eq(hoofdonderwerp)
        if not waar_masker.any():
            raise ValueError(
                f"No validation instances found for {hoofdonderwerp}."
            )
        recalls.append(
            _percentage(
                onderwerp_eval.loc[waar_masker, "main_topic"].eq(
                    hoofdonderwerp
                )
            )
        )
    macro_recall = sum(recalls) / len(recalls)

    sub_end_to_end = _percentage(
        main_correct
        & onderwerp_eval["subtopic"].eq(onderwerp_eval["true_subtopic"])
    )
    sub_bij_juiste_main = _percentage(
        onderwerp_eval.loc[main_correct, "subtopic"].eq(
            onderwerp_eval.loc[main_correct, "true_subtopic"]
        )
    )
    smalltalk_none = _percentage(
        evaluatie.loc[smalltalk, "main_topic"].eq(NONE_LABEL)
    )
    grootste = onderwerp_eval["true_main_topic"].value_counts().max()
    naive_baseline = float(grootste / len(onderwerp_eval) * 100.0)
    voorspelde_onderwerpen = set(
        onderwerp_eval.loc[
            onderwerp_eval["main_topic"].ne(NONE_LABEL), "main_topic"
        ]
    )
    ontbrekende_voorspellingen = set(
        momants_topic.MAIN_TOPICS
    ) - voorspelde_onderwerpen

    metrics = {
        "main_topic_accuracy": hoofd_accuracy,
        "macro_recall": macro_recall,
        "subtopic_accuracy_end_to_end": sub_end_to_end,
        "subtopic_accuracy_given_correct_main": sub_bij_juiste_main,
        "smalltalk_none_accuracy": smalltalk_none,
        "naive_largest_class_baseline": naive_baseline,
        "all_main_topics_predicted": float(not ontbrekende_voorspellingen),
    }

    print("VALIDATION METRICS")
    print(f"Main-topic accuracy (instance-weighted): {hoofd_accuracy:.1f}%")
    print(f"Macro-recall over five main topics: {macro_recall:.1f}%")
    print(f"Subtopic accuracy end-to-end: {sub_end_to_end:.1f}%")
    print(
        "Subtopic accuracy given correct main topic: "
        f"{sub_bij_juiste_main:.1f}%"
    )
    print(f"Smalltalk correctly classified as None: {smalltalk_none:.1f}%")
    print(f"Naive largest-main-topic baseline: {naive_baseline:.1f}%")

    print("\nMAIN-TOPIC CONFUSION MATRIX (rows=true, columns=predicted)")
    matrix = pd.crosstab(
        onderwerp_eval["true_main_topic"],
        onderwerp_eval["main_topic"],
    ).reindex(
        index=momants_topic.MAIN_TOPICS,
        columns=momants_topic.MAIN_TOPICS + [NONE_LABEL],
        fill_value=0,
    )
    print(matrix.to_string())

    ruime_marge = hoofd_accuracy - naive_baseline
    checks = {
        "Main-topic accuracy >= 85%": hoofd_accuracy >= 85.0,
        "Macro-recall >= 75%": macro_recall >= 75.0,
        "Smalltalk None >= 85%": smalltalk_none >= 85.0,
        "All five main topics predicted": not ontbrekende_voorspellingen,
        "Baseline beaten by at least 10 percentage points": ruime_marge >= 10.0,
    }
    print("\nACCEPTANCE CRITERIA")
    for naam, geslaagd in checks.items():
        print(f"- {'PASS' if geslaagd else 'FAIL'}: {naam}")
    if ontbrekende_voorspellingen:
        print(
            "  Missing predicted topics: "
            + ", ".join(sorted(ontbrekende_voorspellingen))
        )

    fout = (
        (
            inhoudelijk
            & (
                evaluatie["main_topic"].ne(evaluatie["true_main_topic"])
                | evaluatie["subtopic"].ne(evaluatie["true_subtopic"])
            )
        )
        | (
            smalltalk
            & evaluatie["main_topic"].ne(NONE_LABEL)
        )
    )
    fouten = evaluatie.loc[
        fout,
        [
            "text",
            "true_main_topic",
            "true_subtopic",
            "main_topic",
            "subtopic",
            "similarity",
        ],
    ].rename(
        columns={
            "true_main_topic": "true_main",
            "true_subtopic": "true_sub",
            "main_topic": "predicted_main",
            "subtopic": "predicted_sub",
        }
    )
    print(f"\nMISCLASSIFIED MESSAGES ({len(fouten)})")
    if fouten.empty:
        print("None")
    else:
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_colwidth",
            None,
            "display.width",
            240,
        ):
            print(fouten.to_string(index=False))
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Momants topic classification against an answer key."
    )
    parser.add_argument("csv_path", type=Path, help="Synthetic Momants test CSV.")
    parser.add_argument("answer_key", type=Path, help="CSV with true topic labels.")
    parser.add_argument(
        "--event-id",
        default=momants_topic.EVENT_ID,
        help="Event ID to select in the v2 topic seed.",
    )
    parser.add_argument(
        "--topics-seed",
        type=Path,
        default=momants_topic.TOPICS_SEED_PATH,
        help="Path to the v2 English topic seed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metrics = valideer(
        csv_pad=args.csv_path,
        antwoordsleutel_pad=args.answer_key,
        seed_pad=args.topics_seed,
        event_id=args.event_id,
        batchgrootte=args.batch_size,
    )
    geslaagd = (
        metrics["main_topic_accuracy"] >= 85.0
        and metrics["macro_recall"] >= 75.0
        and metrics["smalltalk_none_accuracy"] >= 85.0
        and (
            metrics["main_topic_accuracy"]
            - metrics["naive_largest_class_baseline"]
        )
        >= 10.0
        and bool(metrics["all_main_topics_predicted"])
    )
    return 0 if geslaagd else 1


if __name__ == "__main__":
    raise SystemExit(main())