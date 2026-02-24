import matplotlib.pyplot as plt
import pandas as pd

data = [
("R-CNN","Conventional DL",2011),
("SPPNet","Conventional DL",2014),
("Fast R-CNN","Conventional DL",2014),
("Faster R-CNN","Conventional DL",2014),
("FPN","Conventional DL",2017),
("Mask R-CNN","Conventional DL",2017),
("DetectorNet","Hybrid/Bridging",2013),
("PANet","Hybrid/Bridging",2018),
("TridentNet","Hybrid/Bridging",2019),
("SpineNet","Hybrid/Bridging",2021),
("SSD","Advanced DL",2017),
("SqueezeDet","Advanced DL",2018),
("EfficientDet","Advanced DL",2020),
("DETR","Advanced DL",2020),
("YOLO v1","Advanced DL",2015),
("YOLO v2","Advanced DL",2016),
("YOLO v3","Advanced DL",2017),
("YOLO v4","Advanced DL",2018),
("YOLO v5","Advanced DL",2019),
("YOLO v6","Advanced DL",2020),
("YOLO v7","Advanced DL",2021),
("YOLO v8","Advanced DL",2022),
("YOLO v9","Advanced DL",2023),
]

df = pd.DataFrame(data, columns=["System","Category","Year"])

categories = df["Category"].unique()

plt.figure(figsize=(10,6))

for cat in categories:
    subset = df[df["Category"] == cat]
    plt.scatter(subset["Year"], subset["System"], label=cat, s=100)

plt.xlabel("Year")
plt.ylabel("Detection System")
plt.title("Evolution of Object Detection Models")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
