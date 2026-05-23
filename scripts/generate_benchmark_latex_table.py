from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEDUP_KEYS = [
    "dataset",
    "detector_token",
    "tracker",
    "idf1",
    "mota",
    "motp",
    "precision",
    "recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table from the benchmark CSV dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmark_dataset.csv"),
        help="Input CSV dataset path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/report/sections/generated-benchmark-results-table.tex"),
        help="Output .tex file path.",
    )
    return parser.parse_args()


def detector_sort_key(detector_token: str) -> tuple[int, int | str]:
    if detector_token.startswith("yolo"):
        suffix = detector_token[4:]
        return (2, int(suffix) if suffix.isdigit() else suffix)
    return {
        "frcnn": (0, 0),
        "detectron2": (1, 0),
        "detr": (3, 0),
    }.get(detector_token, (9, detector_token))


def format_detector_name(detector_token: str) -> str:
    mapping = {
        "frcnn": "Faster R-CNN",
        "detectron2": "Detectron2",
        "detr": "DETR",
    }
    if detector_token.startswith("yolo"):
        suffix = detector_token[4:]
        return f"YOLOv{suffix}"
    return mapping.get(detector_token, detector_token)


def format_tracker_name(tracker: str) -> str:
    return {
        "sort": "SORT",
        "deepsort": "DeepSORT",
    }.get(tracker, tracker)


def format_metric(value: str) -> str:
    if value == "":
        return "--"
    try:
        return f"{float(value):.3f}"
    except ValueError:
        return value


def load_distinct_rows(input_csv: Path) -> list[dict[str, str]]:
    with input_csv.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    distinct_rows: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(row[column] for column in DEDUP_KEYS)
        if key in seen:
            continue
        seen.add(key)
        distinct_rows.append(row)

    distinct_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["tracker"],
            detector_sort_key(row["detector_token"]),
            row["idf1"],
            row["mota"],
        )
    )
    return distinct_rows


def build_table(rows: list[dict[str, str]]) -> str:
    body_lines = []
    for row in rows:
        detector = format_detector_name(row["detector_token"])
        tracker = format_tracker_name(row["tracker"])
        body_lines.append(
            "\t\t"
            + " & ".join(
                [
                    detector,
                    tracker,
                    format_metric(row["idf1"]),
                    format_metric(row["mota"]),
                    format_metric(row["motp"]),
                    format_metric(row["precision"]),
                    format_metric(row["recall"]),
                ]
            )
            + r"\\"
        )

    lines = [
        r"\begin{table}[th]",
        r"\caption{Distinct benchmark results from the exported KITTI-17 tracking dataset.}",
        r"\label{tab:benchmark-results-distinct}",
        r"\centering",
        r"\begin{NiceTabular}{llrrrrr}",
        r"\CodeBefore",
        r"\rowcolors{2}{gray!10}{white}",
        r"\Body",
        r"\textbf{Detector} & \textbf{Tracker} & \textbf{IDF1} & \textbf{MOTA} & \textbf{MOTP} & \textbf{Precision} & \textbf{Recall} \\",
        *body_lines,
        r"\end{NiceTabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = load_distinct_rows(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_table(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} distinct rows to {args.output}")


if __name__ == "__main__":
    main()
