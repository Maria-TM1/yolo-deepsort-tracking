"""
Generates the evaluation figures from the reported validation metrics.

- ap_by_class_and_size.png: AP@[0.5:0.95] per class and per object size (COCO evaluation)
- class_balance.png: number of annotated instances per class (full dataset)

Run:  python src/plot_ap_by_class_and_size.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

# AP@[0.5:0.95] per class (validation, COCO evaluation)
classes = ["car", "motorcycle", "bicycle", "truck", "traffic cone", "bus", "human"]
ap = [0.618, 0.574, 0.531, 0.501, 0.483, 0.490, 0.438]
order = np.argsort(ap)[::-1]
classes = [classes[i] for i in order]
ap = [ap[i] for i in order]

# AP@[0.5:0.95] per object size (COCO small / medium / large)
ap_size = [0.340, 0.597, 0.756]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))

bars = a1.bar(classes, ap, color="#1f77b4")
a1.set_title("AP@[0.5:0.95] by class")
a1.set_ylabel("AP@[0.5:0.95]")
a1.set_ylim(0, 0.7)
a1.axhline(0.519, ls="--", color="gray", lw=1)
a1.text(len(classes) - 0.5, 0.527, "mAP = 0.519", color="gray", ha="right", fontsize=9)
for bar, value in zip(bars, ap):
    a1.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)
a1.tick_params(axis="x", rotation=35)

a2.bar(["small", "medium", "large"], ap_size, color=["#d62728", "#ff7f0e", "#2ca02c"])
a2.set_title("AP@[0.5:0.95] by object size")
a2.set_ylabel("AP@[0.5:0.95]")
a2.set_ylim(0, 0.8)
for i, value in enumerate(ap_size):
    a2.text(i, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "ap_by_class_and_size.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Instances per class (full dataset)
counts = {
    "bicycle": 11000,
    "bus": 4613,
    "car": 144379,
    "human": 105189,
    "motorcycle": 10346,
    "traffic cone": 52155,
    "truck": 23788,
}
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(list(counts), list(counts.values()), color="steelblue")
ax.set_title("Instances per class (full dataset)")
ax.set_ylabel("Number of instances")
for bar, value in zip(bars, counts.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 1200, f"{value:,}", ha="center", fontsize=10)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(OUT / "class_balance.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Figures saved to {OUT}")
