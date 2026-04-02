# benchmark_plot.py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
print(sorted(fm.get_font_names()))

plt.rcParams["font.family"] = "DejaVu Math TeX Gyre" 

def main():
    # Hardcoded year per detector
    year_by_detector = {
        "Faster R-CNN": 2013,
        "Mask R-CNN": 2017,
        "YOLOv3": 2016,
        "YOLOv5": 2018,
        "YOLOv8": 2021,
        "YOLOv9": 2022,
        "YOLOv10": 2023,
        "YOLOv11": 2024,
        "YOLOv26": 2026,
        "DETR": 2021,
    }

    # Hardcoded benchmark metrics
    metrics = {
            "Faster R-CNN": {"IDF1": 0.711, "MOTA": 0.602, "MOTP": 0.277, "Precision": 0.923, "Recall": 0.671, "Overall": 3.184},
            "Mask R-CNN":   {"IDF1": 0.706, "MOTA": 0.518, "MOTP": 0.211, "Precision": 0.751, "Recall": 0.792, "Overall": 2.849},
            "YOLOv3":       {"IDF1": 0.676, "MOTA": 0.445, "MOTP": 0.210, "Precision": 0.689, "Recall": 0.829, "Overall": 3.178},
            "YOLOv5":       {"IDF1": 0.636, "MOTA": 0.653, "MOTP": 0.224, "Precision": 0.889, "Recall": 0.776, "Overall": 3.378},
            "YOLOv8":       {"IDF1": 0.771, "MOTA": 0.695, "MOTP": 0.218, "Precision": 0.906, "Recall": 0.788, "Overall": 2.806},
            "YOLOv9":       {"IDF1": 0.676, "MOTA": 0.426, "MOTP": 0.200, "Precision": 0.681, "Recall": 0.823, "Overall": 3.248},
            "YOLOv10":      {"IDF1": 0.656, "MOTA": 0.684, "MOTP": 0.219, "Precision": 0.939, "Recall": 0.750, "Overall": 3.239},
            "YOLOv11":      {"IDF1": 0.707, "MOTA": 0.698, "MOTP": 0.221, "Precision": 0.908, "Recall": 0.795, "Overall": 3.329},
            "YOLOv26":      {"IDF1": 0.421, "MOTA": 0.322, "MOTP": 0.247, "Precision": 0.956, "Recall": 0.353, "Overall": 2.299},
            "DETR":         {"IDF1": 0.486, "MOTA": -0.476, "MOTP": 0.228, "Precision": 0.389, "Recall": 0.818, "Overall": 1.445},
    }

    # Pick which metric to plot
    metric_name = "Overall"  # change to "IDF1", "MOTP", "Precision", "Recall"

    # Build x/y series
    points = []
    for detector, vals in metrics.items():
        year = year_by_detector[detector]
        score = vals[metric_name]
        points.append((year, score, detector))

    # Sort by year for a time-aware plot
    points.sort(key=lambda x: x[0])
    years = [p[0] for p in points]
    scores = [p[1] for p in points]
    labels = [p[2] for p in points]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(years, scores, marker="o", linestyle="-", color='#FF5733')
    for x, y, label in points:
        plt.text(x, y, label, fontsize=8, ha="left", va="bottom")

    plt.title(f"{metric_name} by Detector Publication Year")
    plt.xlabel("Publication Year")
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('overall_metric_plot.pdf')

if __name__ == "__main__":
    main()
