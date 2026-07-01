import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

results = [
        {
            "name": "PointRCNN (KITTI)",
            "mAP": "55.12",
            "year": "2018-12-11"
        },
        {
            "name": "PointPillars",
            "mAP": "39.15",
            "year": "2019-5-7"
        },
        {
            "name": "Centerpoint",
            "mAP": "56.92",
            "year": "2021-1-6"
        },
        {
            "name": "SECOND",
            "mAP": "50.9",
            "year": "2018-8-20"
        },
        {
            "name": "RegNet",
            "mAP": "59.21",
            "year": "2020-3-30"
        },
        {
            "name": "VoxelNet (KITTI)",
            "mAP": "56.12",
            "year": "2017-11-17"
        },
        {
            "name": "FocalFormer3D (3)",
            "mAP": "68.7",
            "year": "2023-3-10"
        },
        {
            "name": "TransFusion (28)",
            "mAP": "65.2",
            "year": "2021-8-31"
        },
        {
            "name": "LinK (8)",
            "mAP": "69.8",
            "year": "2022-10-31"
        },
        {
            "name": "Real-Aug++ (2)",
            "mAP": "70.2",
            "year": "2023-5-22"
        },
        {
            "name": "SSN",
            "mAP": "40.92",
            "year": "2020-4-6"
        },
        {
            "name": "VoxelNeXt",
            "mAP": "66.2",
            "year": "2023-3-20"
        }
]
connections = [
    (2, 6),
    (2, 7),
    (2, 8),
    (2, 9),
    (5, 11),
    (1, 2),
    (5, 2)
]


y = [float(result["mAP"]) for result in results]
x = [datetime.strptime(result["year"], '%Y-%m-%d').date() for result in results]
fig, ax = plt.subplots(figsize=(20, 10))
ax.scatter(x,
           y,
           color="blue",
           marker="o",
           s=100)
for i, result in enumerate(results):
    ax.annotate(
            result["name"],
            (datetime.strptime(result["year"], "%Y-%m-%d").date(), float(result["mAP"])),
            textcoords="offset points",
            xytext=(10, 5),
            ha="left",
            va="bottom",
            fontsize=10
        )

for i, j in connections:
    ax.plot([x[i], x[j]], 
            [y[i], y[j]],
            color="grey",
            alpha=0.4)
ax.set_xlabel("Year")
ax.set_ylabel("mAP Score")
plt.savefig("detector-mAP-dashboard.png")
