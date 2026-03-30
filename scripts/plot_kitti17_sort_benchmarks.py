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
        "IDF1": 0.711,
        "MOTA": 0.602,
        "MOTP": 0.277,
        "Precision": 0.923,
        "Recall": 0.671,
    },
    {
        "detector": "Mask R-CNN",
        "year": 2017,
        "IDF1": 0.706,
        "MOTA": 0.518,
        "MOTP": 0.211,
        "Precision": 0.751,
        "Recall": 0.792,
    },
    {
        "detector": "YOLOv3",
        "year": 2016,
        "IDF1": 0.676,
        "MOTA": 0.445,
        "MOTP": 0.210,
        "Precision": 0.689,
        "Recall": 0.829,
    },
    {
        "detector": "YOLOv5",
        "year": 2018,
        "IDF1": 0.636,
        "MOTA": 0.653,
        "MOTP": 0.224,
        "Precision": 0.889,
        "Recall": 0.776,
    },
    {
        "detector": "YOLOv8",
        "year": 2021,
        "IDF1": 0.771,
        "MOTA": 0.695,
        "MOTP": 0.218,
        "Precision": 0.906,
        "Recall": 0.788,
    },
    {
        "detector": "YOLOv9",
        "year": 2022,
        "IDF1": 0.676,
        "MOTA": 0.426,
        "MOTP": 0.200,
        "Precision": 0.681,
        "Recall": 0.823,
    },
    {
        "detector": "YOLOv10",
        "year": 2023,
        "IDF1": 0.656,
        "MOTA": 0.684,
        "MOTP": 0.219,
        "Precision": 0.939,
        "Recall": 0.750,
    },
    {
        "detector": "YOLOv11",
        "year": 2024,
        "IDF1": 0.707,
        "MOTA": 0.698,
        "MOTP": 0.221,
        "Precision": 0.908,
        "Recall": 0.795,
    },
    {
        "detector": "YOLOv26",
        "year": 2025,
        "IDF1": 0.421,
        "MOTA": 0.322,
        "MOTP": 0.247,
        "Precision": 0.956,
        "Recall": 0.353,
    },
    {
        "detector": "DETR",
        "year": 2021,
        "IDF1": 0.684,
        "MOTA": 0.412,
        "MOTP": 0.228,
        "Precision": 0.673,
        "Recall": 0.816,
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
    return OUTPUT_DIR / f"kitti17_sort_benchmarks_by_year_{metric.lower()}.png"


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
            if row["detector"] is "DETR" or row["detector"] is "Mask R-CNN":
                ax.annotate(
                    row["detector"],
                    (row["year"], row[metric]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    fontsize=12,
                )
            elif row["detector"] is "YOLOv26":
                ax.annotate(
                    row["detector"],
                    (row["year"], row[metric]),
                    textcoords="offset points",
                    xytext=(-23, -15),
                    fontsize=12,
                )
            else:
                ax.annotate(
                    row["detector"],
                    (row["year"], row[metric]),
                    textcoords="offset points",
                    xytext=(0, -15),
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
        ax.legend()
        ax.set_ylim(0,1)
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
