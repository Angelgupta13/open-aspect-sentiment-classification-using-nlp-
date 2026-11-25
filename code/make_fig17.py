import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8.5,
                      "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                      "savefig.bbox": "tight", "savefig.pad_inches": 0.04})

datasets = ["2014\nRestaurant", "2016 Restaurant\n(explicit targets)", "2016 Restaurant\n(category-level)",
            "2014\nLaptop", "2016 Laptop\n(category-level)"]
nli   = [0.741, 0.739, 0.674, 0.691, 0.597]
cloze = [0.548, 0.547, 0.572, 0.583, 0.495]
both  = [0.619, 0.723, 0.709, 0.633, 0.603]

x = np.arange(len(datasets))
w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.bar(x - w, nli, w, label="NLI", color="#4C72B0")
ax.bar(x, cloze, w, label="Cloze", color="#DD8452")
ax.bar(x + w, both, w, label="Both", color="#55A868")
ax.set_ylabel("Zero-shot macro-F1")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 0.85)
ax.axvline(2.63, color="0.7", lw=0.8, ls="--")
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
for i, v in enumerate(nli):
    ax.text(x[i]-w, v+0.015, f"{v:.2f}", ha="center", fontsize=6.5)
for i, v in enumerate(cloze):
    ax.text(x[i], v+0.015, f"{v:.2f}", ha="center", fontsize=6.5)
for i, v in enumerate(both):
    ax.text(x[i]+w, v+0.015, f"{v:.2f}", ha="center", fontsize=6.5)
fig.tight_layout()
fig.savefig("/home/claude/work/sem16/figs/fig17_sem16_zeroshot.png", dpi=300)
fig.savefig("/home/claude/work/sem16/figs/fig17_sem16_zeroshot.pdf")
print("saved")
