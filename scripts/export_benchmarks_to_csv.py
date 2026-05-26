from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


RUN_FILE_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})-(?P<dataset>.+)-(?P<detector_token>[^-]+)-(?P<tracker>[^-]+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export benchmark text summaries to a CSV dataset."
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        dest="input_dirs",
        help="Directory containing benchmark .txt files. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file path.",
    )
    return parser.parse_args()


def infer_run_metadata(file_path: Path) -> dict[str, str]:
    match = RUN_FILE_PATTERN.match(file_path.stem)
    if not match:
        return {
            "run_timestamp": "",
            "dataset_from_filename": "",
            "detector_token": "",
            "tracker_from_filename": "",
            "detector_family": "",
            "detector_variant": "",
        }

    detector_token = match.group("detector_token")
    detector_family = "yolo" if detector_token.startswith("yolo") else detector_token
    detector_variant = detector_token if detector_token.startswith("yolo") else ""

    return {
        "run_timestamp": match.group("timestamp"),
        "dataset_from_filename": match.group("dataset"),
        "detector_token": detector_token,
        "tracker_from_filename": match.group("tracker"),
        "detector_family": detector_family,
        "detector_variant": detector_variant,
    }


def convert_value(raw_value: str) -> int | float | str:
    lowered = raw_value.lower()
    if lowered == "nan":
        return ""

    try:
        numeric_value = float(raw_value)
    except ValueError:
        return raw_value

    if math.isfinite(numeric_value) and numeric_value.is_integer():
        return int(numeric_value)

    return numeric_value


def parse_benchmark_file(file_path: Path) -> dict[str, int | float | str]:
    lines = [line.rstrip("\n") for line in file_path.read_text(encoding="utf-8").splitlines()]

    metadata: dict[str, str] = {}
    for line in lines[:4]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()

    content_lines = [line for line in lines[4:] if line.strip()]
    if len(content_lines) < 2:
        raise ValueError(f"Benchmark file {file_path} does not contain a metrics table.")

    header_tokens = content_lines[0].split()
    value_tokens = content_lines[1].split()
    if len(value_tokens) != len(header_tokens) + 1:
        raise ValueError(
            f"Unexpected metrics row format in {file_path}: "
            f"{len(header_tokens)} headers, {len(value_tokens)} values."
        )

    row: dict[str, int | float | str] = {
        "source_file": str(file_path.relative_to(file_path.parents[1])),
        "sequence_name": value_tokens[0],
        "timestamp": metadata.get("timestamp", ""),
        "dataset": metadata.get("dataset", ""),
        "detector": metadata.get("detector", ""),
        "tracker": metadata.get("tracker", ""),
    }
    row.update(infer_run_metadata(file_path))

    for key, raw_value in zip(header_tokens, value_tokens[1:]):
        row[key] = convert_value(raw_value)

    return row


def collect_benchmark_files(input_dirs: list[Path]) -> list[Path]:
    benchmark_files: list[Path] = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            continue
        benchmark_files.extend(sorted(input_dir.rglob("*.txt")))
    return benchmark_files


def export_benchmarks(input_dirs: list[Path], output_csv: Path) -> int:
    benchmark_files = collect_benchmark_files(input_dirs)
    rows = [parse_benchmark_file(file_path) for file_path in benchmark_files]
    if not rows:
        searched_dirs = ", ".join(str(path) for path in input_dirs)
        raise ValueError(f"No benchmark .txt files found in: {searched_dirs}")

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    input_dirs = (
        [Path(input_dir) for input_dir in args.input_dirs]
        if args.input_dirs
        else [project_root / "benchmarks", project_root / "data" / "benchmark"]
    )
    output_csv = args.output or (project_root / "data" / "benchmark_dataset.csv")
    row_count = export_benchmarks(input_dirs, output_csv)
    print(f"Exported {row_count} benchmark runs to {output_csv}")


if __name__ == "__main__":
    main()
