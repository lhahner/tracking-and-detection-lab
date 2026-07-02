import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

results = [
        {
            "name": "PointRCNN (KITTI)",
            "mAP": "55.12", # custom - not implemented
            "year": "2018-12-11",
            "implemented": False
        },
        {
            "name": "PointPillars",
            "mAP": "39.7", # mmdetection3d - implemented 
            "year": "2019-5-7",
            "implemented": True
        },
        {
            "name": "Centerpoint-Pointpillar",
            "mAP": "48.72", # mmdetection3d - implemented 
            "year": "2021-1-6",
            "implemented": True
        },
        {
            "name": "Centerpoint-Voxelnet",
            "mAP": "56.11", # mmdetection3d - implemented
            "year": "2021-1-6",
            "implemented": True
        },
        {
            "name": "SECOND",
            "mAP": "50.6", # OpenPCDet - Implemented
            "year": "2018-8-20",
            "implemented": True
        },
        {
            "name": "RegNet",
            "mAP": "59.21", # mmdetection3d - implemeted
            "year": "2020-3-30",
            "implemented": True
        },
        {
            "name": "VoxelNet (KITTI)",
            "mAP": "56.12", # custom - not implemented
            "year": "2017-11-17",
            "implemented": False
        },
        {
            "name": "FocalFormer3D (3)",
            "mAP": "68.7", # custom - not implemented
            "year": "2023-3-10",
            "implemented": False
        },
        {
            "name": "TransFusion (28)",
            "mAP": "64.58", # openpcdet - not implemented 
            "year": "2021-8-31",
            "implemented": False
        },
        {
            "name": "LinK (8)",
            "mAP": "69.8", # custom - not implemented
            "year": "2022-10-31",
            "implemented": False
        },
        {
            "name": "Real-Aug++ (2)",
            "mAP": "70.2", # custom - not implemented
            "year": "2023-5-22",
            "implemented": False
        },
        {
            "name": "SSN",
            "mAP": "40.92", # mmdetection3d - implemented
            "year": "2020-4-6",
            "implemented": True
        },
        {
            "name": "VoxelNeXt",
            "mAP": "66.2", # openpcdet - not implemented
            "year": "2023-3-20",
            "implemented": False
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
