from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 18

BENCHMARK_DATA = [
    {
        "detector": "Faster R-CNN",
        "year": 2014,
        "IDF1": 0.306,
        "MOTA": 0.349,
        "MOTP": 0.231,
        "Precision": 0.955,
        "Recall": 0.381,
    },
    {
        "detector": "Mask R-CNN",
        "year": 2017,
        "IDF1": 0.270,
        "MOTA": 0.284,
        "MOTP": 0.216,
        "Precision": 0.959,
        "Recall": 0.309,
    },
    {
        "detector": "YOLOv3",
        "year": 2016,
        "IDF1": 0.361,
        "MOTA": 0.354,
        "MOTP": 0.210,
        "Precision": 0.970,
        "Recall": 0.375,
    },
    {
        "detector": "YOLOv5",
        "year": 2018,
        "IDF1": 0.247,
        "MOTA": 0.285,
        "MOTP": 0.251,
        "Precision": 0.934,
        "Recall": 0.323,
    },
    {
        "detector": "YOLOv8",
        "year": 2021,
        "IDF1": 0.291,
        "MOTA": 0.313,
        "MOTP": 0.242,
        "Precision": 0.948,
        "Recall": 0.348,
    },
    {
        "detector": "YOLOv9",
        "year": 2022,
        "IDF1": 0.366,
        "MOTA": 0.362,
        "MOTP": 0.214,
        "Precision": 0.959,
        "Recall": 0.389,
    },
    {
        "detector": "YOLOv10",
        "year": 2023,
        "IDF1": 0.144,
        "MOTA": 0.160,
        "MOTP": 0.216,
        "Precision": 0.983,
        "Recall": 0.174,
    },
    {
        "detector": "YOLOv11",
        "year": 2024,
        "IDF1": 0.275,
        "MOTA": 0.255,
        "MOTP": 0.222,
        "Precision": 0.977,
        "Recall": 0.272,
    },
    {
        "detector": "YOLOv26",
        "year": 2025,
        "IDF1": 0.177,
        "MOTA": 0.178,
        "MOTP": 0.210,
        "Precision": 0.987,
        "Recall": 0.190,
    },
    {
        "detector": "DETR",
        "year": 2021,
        "IDF1": 0.190,
        "MOTA": 0.169,
        "MOTP": 0.201,
        "Precision": 0.811,
        "Recall": 0.238,
    },
]

METRICS = ["MOTA", "MOTP"]
OUTPUT_DIR = Path(__file__).parent


def build_dataframe() -> pd.DataFrame:
    """Build a sorted benchmark DataFrame from the static dataset.

    Returns:
        pd.DataFrame: Benchmark rows sorted by year and detector name.
    """
    return pd.DataFrame(BENCHMARK_DATA).sort_values(["year", "detector"]).reset_index(
        drop=True
    )


def output_path_for_metric(metric: str) -> Path:
    """Return the output image path for a benchmark metric plot.

    Args:
        metric: Metric name used in the filename.

    Returns:
        Path: Output path for the saved figure.
    """
    return OUTPUT_DIR / f"mot20_01_deepsort_benchmarks_by_year_{metric.lower()}.png"


def plot_metrics(df: pd.DataFrame) -> list[Path]:
    """Create and save one scatter plot per metric.

    Args:
        df: Benchmark DataFrame to plot.

    Returns:
        list[Path]: Paths to the generated figure files.
    """
    output_paths: list[Path] = []

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df["year"], df[metric], s=90, color="tab:green")

        for _, row in df.iterrows():
            ax.annotate(
                row["detector"],
                (row["year"], row[metric]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=12,
            )

        yearly_mean = df.groupby("year", as_index=False)[metric].mean()
        ax.plot(
            yearly_mean["year"],
            yearly_mean[metric],
            linestyle="--",
            color="dimgray",
            linewidth=1.2,
            label="Yearly mean",
        )

        ax.set_title(f"")
        ax.set_xlabel("Publication Year")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0,1)
        ax.legend()

        output_path = output_path_for_metric(metric)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        output_paths.append(output_path)
        plt.close(fig)

    return output_paths


def main() -> None:
    """Build the benchmark table, render plots, and print saved paths."""
    df = build_dataframe()
    print(df.to_string(index=False))
    output_paths = plot_metrics(df)
    print("\nSaved figures to:")
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
