import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import json

from pathlib import Path
from datetime import datetime

dashboard_results_path = Path("inputs/dashboard_data_implemented_only.json")
with open(dashboard_results_path, "r") as file:
    results = json.load(file)

results = sorted(results,
              key=lambda r: datetime.strptime(r["year"], "%Y-%m-%d")
)
connections = [
    (1, 2),
    (2, 7),
    (2, 8),
    (2, 9),
    (6, 11)
]

# Gathering the data
mota = [float(result["mota_simple_track"]) for result in results]
motp = [float(result["motp_simple_track"]) for result in results]
mAP_baseline = [float(result["mAP_baseline"]) for result in results]
mAP = [float(result["mAP"]) for result in results]
date = [datetime.strptime(result["year"], '%Y-%m-%d').date() for result in results]

# Building the plots
fig, ax = plt.subplots(figsize=(20, 10))
mota_mop_axis = ax.twinx()

ax.set_zorder(2)
mota_mop_axis.set_zorder(1)
ax.patch.set_visible(False)

date_positions = mdates.date2num(date) 



bar_width = 25
bar_offset = 18

mota_bars = mota_mop_axis.bar(
        date_positions - bar_offset,
        mota,
        width=bar_width,
        alpha=0.8,
        label="MOTA",
        zorder=1
)
motp_bars = mota_mop_axis.bar(
        date_positions + bar_offset,
        motp,
        width=bar_width,
        alpha=0.8,
        label="MOTP",
        zorder=1
)
ax.plot(
    date,
    mAP,
    label="mAP",
    zorder=5
)
ax.plot(
        date,
        mAP_baseline,
        label="mAP baseline",
        zorder=5)

# Annotation
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
    ax.annotate(
            result["name"],
            (datetime.strptime(result["year"], "%Y-%m-%d").date(), float(result["mAP_baseline"])),
            textcoords="offset points",
            xytext=(10, 5),
            ha="left",
            va="bottom",
            fontsize=10
        )
#for i, j in connections:
#    ax.plot([date[i], mAP[j]], 
#            [mAP[i], mAP[j]],
#            color="grey",
#            alpha=0.4)
ax.set_xlabel("Year")
ax.set_ylabel("mAP Score")
mota_mop_axis.set_ylabel("MOTA/MOTP Score")
ax.xaxis_date()
ax.xaxis.set_major_locator(
        mdates.YearLocator()
)
ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
)
ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
)

handles_1, labels_1 = ax.get_legend_handles_labels()
handles_2, labels_2 = mota_mop_axis.get_legend_handles_labels()

ax.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="upper left"
)

fig.autofmt_xdate()
fig.tight_layout()
plt.savefig("detector-mAP-dashboard.png")
