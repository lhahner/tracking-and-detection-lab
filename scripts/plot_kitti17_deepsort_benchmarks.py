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
        "IDF1": 0.500,
        "MOTA": 0.367,
        "MOTP": 0.275,
        "Precision": 0.918,
        "Recall": 0.410,
    },
    {
        "detector": "Mask R-CNN",
        "year": 2017,
        "IDF1": 0.668,
        "MOTA": 0.425,
        "MOTP": 0.211,
        "Precision": 0.708,
        "Recall": 0.748,
    },
    {
        "detector": "YOLOv3",
        "year": 2016,
        "IDF1": 0.638,
        "MOTA": 0.346,
        "MOTP": 0.208,
        "Precision": 0.654,
        "Recall": 0.770,
    },
    {
        "detector": "YOLOv5",
        "year": 2018,
        "IDF1": 0.706,
        "MOTA": 0.556,
        "MOTP": 0.225,
        "Precision": 0.832,
        "Recall": 0.717,
    },
    {
        "detector": "YOLOv8",
        "year": 2021,
        "IDF1": 0.658,
        "MOTA": 0.587,
        "MOTP": 0.224,
        "Precision": 0.865,
        "Recall": 0.713,
    },
    {
        "detector": "YOLOv9",
        "year": 2022,
        "IDF1": 0.637,
        "MOTA": 0.329,
        "MOTP": 0.209,
        "Precision": 0.647,
        "Recall": 0.757,
    },
    {
        "detector": "YOLOv10",
        "year": 2023,
        "IDF1": 0.673,
        "MOTA": 0.602,
        "MOTP": 0.226,
        "Precision": 0.903,
        "Recall": 0.697,
    },
    {
        "detector": "YOLOv11",
        "year": 2024,
        "IDF1": 0.682,
        "MOTA": 0.608,
        "MOTP": 0.222,
        "Precision": 0.869,
        "Recall": 0.735,
    },
    {
        "detector": "YOLOv26",
        "year": 2025,
        "IDF1": 0.376,
        "MOTA": 0.252,
        "MOTP": 0.228,
        "Precision": 0.941,
        "Recall": 0.283,
    },
    {
        "detector": "DETR",
        "year": 2021,
        "IDF1": 0.607,
        "MOTA": 0.295,
        "MOTP": 0.237,
        "Precision": 0.640,
        "Recall": 0.717,
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
    return OUTPUT_DIR / f"kitti17_deepsort_benchmarks_by_year_{metric.lower()}.png"


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
            if row["detector"] is "Mask R-CNN" or row["detector"] is "YOLOv8" or row["detector"] is "YOLOv11":
                    ax.annotate(
                        row["detector"],
                        (row["year"], row[metric]),
                        textcoords="offset points",
                        xytext=(6, -17),
                        fontsize=12,
                    )
            elif row["detector"] is "YOLOv26":
                ax.annotate(
                    row["detector"],
                    (row["year"], row[metric]),
                    textcoords="offset points",
                    xytext=(-25, 4),
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
        ax.set_ylim(0,1)
        ax.set_xlabel("Publication Year")
        ax.set_ylabel(metric)
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
