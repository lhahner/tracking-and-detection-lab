from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 18

BENCHMARK_DATA = [
    {
        "detector": "Faster R-CNN",
        "year": 2014,
        "IDF1": 0.358,
        "MOTA": 0.349,
        "MOTP": 0.224,
        "Precision": 0.966,
        "Recall": 0.371,
    },
    {
        "detector": "Mask R-CNN",
        "year": 2017,
        "IDF1": 0.307,
        "MOTA": 0.284,
        "MOTP": 0.205,
        "Precision": 0.976,
        "Recall": 0.298,
    },
    {
        "detector": "YOLOv3",
        "year": 2016,
        "IDF1": 0.388,
        "MOTA": 0.347,
        "MOTP": 0.202,
        "Precision": 0.977,
        "Recall": 0.362,
    },
    {
        "detector": "YOLOv5",
        "year": 2018,
        "IDF1": 0.260,
        "MOTA": 0.276,
        "MOTP": 0.242,
        "Precision": 0.950,
        "Recall": 0.305,
    },
    {
        "detector": "YOLOv8",
        "year": 2021,
        "IDF1": 0.298,
        "MOTA": 0.306,
        "MOTP": 0.232,
        "Precision": 0.965,
        "Recall": 0.329,
    },
    {
        "detector": "YOLOv9",
        "year": 2022,
        "IDF1": 0.418,
        "MOTA": 0.361,
        "MOTP": 0.207,
        "Precision": 0.972,
        "Recall": 0.378,
    },
    {
        "detector": "YOLOv10",
        "year": 2023,
        "IDF1": 0.159,
        "MOTA": 0.145,
        "MOTP": 0.199,
        "Precision": 0.990,
        "Recall": 0.154,
    },
    {
        "detector": "YOLOv11",
        "year": 2024,
        "IDF1": 0.274,
        "MOTA": 0.247,
        "MOTP": 0.211,
        "Precision": 0.988,
        "Recall": 0.257,
    },
    {
        "detector": "YOLOv26",
        "year": 2025,
        "IDF1": 0.196,
        "MOTA": 0.165,
        "MOTP": 0.195,
        "Precision": 0.991,
        "Recall": 0.173,
    },
    {
        "detector": "DETR",
        "year": 2021,
        "IDF1": 0.220,
        "MOTA": 0.154,
        "MOTP": 0.190,
        "Precision": 0.794,
        "Recall": 0.219,
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
    return OUTPUT_DIR / f"mot20_01_sort_benchmarks_by_year_{metric.lower()}.png"


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
        ax.scatter(df["year"], df[metric], s=90, color="tab:red")

        for _, row in df.iterrows():
            if row["detector"] is "YOLOv26":
                ax.annotate(
                    row["detector"],
                    (row["year"], row[metric]),
                    textcoords="offset points",
                    xytext=(-22, -12),
                    fontsize=12,
                )
            else:
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
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
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
