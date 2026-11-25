"""Paper figures for the ATSC prompting project, built from the saved result files.
Run:  python make_figures.py   (expects atsc_results.csv, objective2_results.csv, cot_flips.txt alongside)
Writes PNG (300 dpi) + PDF (vector) for every figure into ./figs
"""
import re, os
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "figs"; os.makedirs(OUT, exist_ok=True)

# ---- style: one palette, fixed role per method everywhere --------------------
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
C = {"prompt-nli": "#2a78d6", "prompt-cloze": "#eb6834", "prompt-both": "#1baf7a", "bert-head": "#3d3c39"}
NAME = {"prompt-nli": "Prompt-NLI", "prompt-cloze": "Prompt-Cloze", "prompt-both": "Prompt-NLI+Cloze",
        "bert-head": "BERT fine-tuned"}
MARK = {"prompt-nli": "o", "prompt-cloze": "s", "prompt-both": "^", "bert-head": "D"}
CLS = ["positive", "negative", "neutral"]
CLS_C = ["#1baf7a", "#e34948", "#8a8984"]

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300); fig.savefig(f"{OUT}/{name}.pdf"); plt.close(fig)

R = pd.read_csv("atsc_results.csv")
O2 = pd.read_csv("objective2_results.csv")

# =============================================================================
# Fig 1  Method overview (schematic)
# =============================================================================
def box(ax, x, y, w, h, text, fc="#ffffff", ec=MUTED, fs=8, bold=False, tc=INK, lw=1.0, mono=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3,
            fontweight="bold" if bold else "normal", family="DejaVu Sans Mono" if mono else "DejaVu Sans",
            linespacing=1.35)
def arrow(ax, x0, y0, x1, y1, color=INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 lw=1.1, color=color, zorder=1, shrinkA=0, shrinkB=0))

fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_xlim(0, 144); ax.set_ylim(0, 66); ax.axis("off")
ax.set_aspect("equal")
MONO = "DejaVu Sans Mono"
# input
box(ax, 0, 22, 25, 22, "", ec=MUTED)
ax.text(12.5, 40.5, "Input", ha="center", fontsize=7.5, fontweight="bold", color=INK)
ax.text(12.5, 32.5, "\u201cGreat food,\nbut the service\nwas slow.\u201d", ha="center", va="center", fontsize=6.8,
        color=INK, style="italic")
ax.text(12.5, 25, "aspect = service", ha="center", fontsize=6.5, color=INK2, family=MONO)
# NLI branch
box(ax, 33, 38, 44, 26, "", fc="#eaf2fc", ec=C["prompt-nli"])
ax.text(55, 60, "NLI branch", ha="center", fontsize=7.8, color=C["prompt-nli"], fontweight="bold")
ax.text(55, 56.3, "RoBERTa-large-MNLI (frozen)", ha="center", fontsize=6.3, color=INK2)
ax.text(36, 52, "premise: <review>", fontsize=6.1, family=MONO, color=INK, va="center")
for k, w in enumerate(["good", "bad", "okay"]):
    ax.text(36, 48 - 3.4*k, f"hyp {k+1}: The service is {w}.", fontsize=6.1, family=MONO, color=INK, va="center")
# Cloze branch
box(ax, 33, 2, 44, 26, "", fc="#fdeee7", ec=C["prompt-cloze"])
ax.text(55, 24, "Cloze branch", ha="center", fontsize=7.8, color=C["prompt-cloze"], fontweight="bold")
ax.text(55, 20.3, "RoBERTa-large MLM (frozen)", ha="center", fontsize=6.3, color=INK2)
ax.text(35, 16, "<review> The service is <mask>.", fontsize=5.9, family=MONO, color=INK, va="center")
ax.text(36, 11, "pos: good | great", fontsize=6.1, family=MONO, color=INK, va="center")
ax.text(36, 7.6, "neg: bad  | terrible", fontsize=6.1, family=MONO, color=INK, va="center")
ax.text(36, 4.2, "neu: okay | fine", fontsize=6.1, family=MONO, color=INK, va="center")
arrow(ax, 25.5, 36, 32.5, 48); arrow(ax, 25.5, 30, 32.5, 18)
# score bars
def bars(x, y, vals, color, label):
    w, g, H = 3.4, 1.0, 13
    for i, v in enumerate(vals):
        ax.add_patch(plt.Rectangle((x + i*(w+g), y), w, H*v, fc=CLS_C[i], ec="none", zorder=3))
        ax.text(x + i*(w+g) + w/2, y - 1.0, ["pos", "neg", "neu"][i], ha="center", va="top", fontsize=5.6, color=INK2)
    ax.plot([x - 0.8, x + 3*w + 2*g + 0.8], [y, y], color=MUTED, lw=0.6)
    ax.text(x + (3*w + 2*g)/2, y + H + 1.5, label, ha="center", fontsize=6.2, color=color, linespacing=1.2)
bars(80, 42, [0.15, 0.95, 0.30], C["prompt-nli"], "3 entailment\nscores")
bars(80, 8, [0.45, 0.55, 0.40], C["prompt-cloze"], "3 label-word\nscores")
arrow(ax, 77.3, 51, 79.5, 51); arrow(ax, 77.3, 15, 79.5, 15)
ax.text(86.7, 33, "log-softmax\nover 3 classes", ha="center", va="center", fontsize=5.8, color=MUTED)
# heads
box(ax, 99, 39, 22, 17, "", ec=INK2)
ax.text(110, 51.5, "Zero-shot", ha="center", fontsize=7.2, fontweight="bold", color=INK)
ax.text(110, 44.5, "argmax of scores\n(sum if both)\n0 labels", ha="center", va="center", fontsize=6, color=INK2)
box(ax, 99, 10, 22, 19, "", ec=INK2)
ax.text(110, 24.5, "Few-shot", ha="center", fontsize=7.2, fontweight="bold", color=INK)
ax.text(110, 16.5, "logistic regression\non 3 (or 6 concat)\nscores\nn = 16 \u2026 full", ha="center",
        va="center", fontsize=6, color=INK2)
arrow(ax, 94.5, 50, 98.5, 48); arrow(ax, 94.5, 47, 98.5, 22)
arrow(ax, 94.5, 17, 98.5, 44); arrow(ax, 94.5, 14, 98.5, 18)
# output
box(ax, 126, 25, 18, 16, "", fc="#f4f4f2", ec=INK2)
ax.text(135, 37, "Polarity", ha="center", fontsize=7.2, fontweight="bold", color=INK)
for i, c in enumerate(CLS):
    ax.add_patch(plt.Rectangle((128.5, 32.2 - 3.3*i), 1.8, 1.8, fc=CLS_C[i], ec="none"))
    ax.text(131.5, 33.1 - 3.3*i, c, fontsize=6, va="center", color=INK)
arrow(ax, 121.3, 47, 125.6, 37); arrow(ax, 121.3, 20, 125.6, 29)
ax.text(72, -3.5, "Baseline: BERT-base fine-tuned end-to-end on (review, aspect) pairs.   Objective 2: identical "
        "prompts with an aspect category (e.g. \u201cprice\u201d) in place of the term.", ha="center", fontsize=6,
        color=INK2, style="italic")
save(fig, "fig1_method_overview")

# =============================================================================
# Fig 2  Dataset label distribution
# =============================================================================
counts = {"Restaurant train": [2164, 805, 633], "Restaurant test": [728, 196, 196],
          "Laptop train": [987, 866, 460], "Laptop test": [341, 128, 169]}
fig, ax = plt.subplots(figsize=(3.5, 1.9))
names = list(counts)[::-1]
for j, k in enumerate(names):
    v = np.array(counts[k]); p = v / v.sum(); left = 0
    for i in range(3):
        ax.barh(j, p[i], left=left, color=CLS_C[i], height=0.62, edgecolor="white", linewidth=1.5)
        if p[i] > 0.09:
            ax.text(left + p[i]/2, j, f"{p[i]*100:.0f}%", ha="center", va="center", fontsize=6.8, color="white",
                    fontweight="bold")
        left += p[i]
    ax.text(1.01, j, f"n={v.sum()}", va="center", fontsize=7, color=INK2)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
ax.grid(axis="y", visible=False)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in CLS_C], CLS, ncol=3, loc="lower center",
          bbox_to_anchor=(0.45, 1.0), fontsize=7, handlelength=1, columnspacing=1)
save(fig, "fig2_label_distribution")

# =============================================================================
# Fig 3  Label-efficiency learning curves, 4 transfer settings
# =============================================================================
zs = {("R",): None}
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.35), sharey=True)
for ax, st in zip(axs, ["R->R", "L->L", "R->L", "L->R"]):
    tgt = st[-1]
    for m in ["prompt-nli", "prompt-both", "prompt-cloze"]:
        d = R[(R.method == m) & (R.setting == st)].sort_values("n")
        ax.plot(d.n, d.f1, color=C[m], marker=MARK[m], ms=4, lw=1.6, label=NAME[m])
        z = R[(R.method == m) & (R.setting == f"{tgt} (zero-shot)")].f1.item()
        ax.axhline(z, color=C[m], lw=1, ls=(0, (3, 2)), alpha=0.9)
    b = R[(R.method == "bert-head") & (R.setting == st)].sort_values("n")
    ax.plot(b.n, b.f1, color=C["bert-head"], marker="D", ms=4.5, lw=1.2, ls=(0, (1, 1.5)), label=NAME["bert-head"])
    ax.set_xscale("log"); ax.set_xticks([16, 64, 256, 1024]); ax.set_xticklabels(["16", "64", "256", "1k"])
    ax.minorticks_off()
    src, dst = {"R": "Rest.", "L": "Lap."}[st[0]], {"R": "Rest.", "L": "Lap."}[st[-1]]
    ax.set_title(f"{src} → {dst}" + ("  (cross)" if st[0] != st[-1] else ""), fontsize=9)
    xs_ = sorted(R[(R.setting == st) & (R.method == "prompt-nli")].n)
    ax.set_xticks([16, 64, 256, 1024, xs_[-1]]); ax.set_xticklabels(["16", "64", "256", "1k", "full"], fontsize=7)
axs[0].set_ylabel("macro-F1"); axs[0].set_ylim(0.48, 0.78)
h, l = axs[0].get_legend_handles_labels()
h.append(mpl.lines.Line2D([], [], color=MUTED, lw=1, ls=(0, (3, 2)))); l.append("zero-shot (same colour)")
fig.legend(h, l, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.09), fontsize=7.2, handlelength=2.2)
fig.supxlabel("labelled training examples (log scale)", fontsize=9, y=-0.06)
fig.text(0.995, -0.13, "Prompt models: mean of 5 random draws per n (std not saved). BERT: single seed. "
         "Rightmost point = full source training set.", ha="right", fontsize=6, color=MUTED)
save(fig, "fig3_learning_curves")

# =============================================================================
# Fig 4  Domain shift: in-domain vs cross-domain on the SAME test set
# =============================================================================
def f1(m, st, n):
    d = R[(R.method == m) & (R.setting == st)]
    if n == "full": d = d[d.n == d.n.max()]
    else: d = d[d.n == n]
    return d.f1.item()
rows = []
for tgt, ind, crs in [("Restaurant test", "R->R", "L->R"), ("Laptop test", "L->L", "R->L")]:
    for m, n, lab in [("prompt-nli", 64, "NLI, n=64"), ("prompt-nli", "full", "NLI, full"),
                      ("prompt-both", "full", "NLI+Cloze, full"),
                      ("bert-head", 256, "BERT, n=256"), ("bert-head", "full", "BERT, full")]:
        rows.append((tgt, lab, m, f1(m, ind, n), f1(m, crs, n)))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3), sharex=True)
for ax, tgt in zip(axs, ["Restaurant test", "Laptop test"]):
    rr = [r for r in rows if r[0] == tgt][::-1]
    for j, (_, lab, m, a, b) in enumerate(rr):
        ax.plot([a, b], [j, j], color=GRID, lw=3.5, solid_capstyle="round", zorder=1)
        ax.scatter(a, j, s=34, color=C[m], zorder=3, edgecolor="white", lw=1.2)
        ax.scatter(b, j, s=34, facecolor="white", edgecolor=C[m], lw=1.6, zorder=3)
        d = b - a
        ax.text(max(a, b) + 0.008, j, f"{d:+.3f}", va="center", fontsize=7,
                color="#c23b3a" if d < -0.02 else INK2, fontweight="bold" if d < -0.02 else "normal")
    ax.set_yticks(range(len(rr))); ax.set_yticklabels([r[1] for r in rr]); ax.grid(axis="y", visible=False)
    ax.set_title(tgt, fontsize=9); ax.set_xlabel("macro-F1")
axs[0].set_xlim(0.49, 0.80)
fig.legend([mpl.lines.Line2D([], [], marker="o", ls="", color=INK2, ms=5),
            mpl.lines.Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=5)],
           ["trained in-domain", "trained on other domain"], ncol=2, loc="upper center",
           bbox_to_anchor=(0.5, 1.07), fontsize=7.2)
fig.text(0.995, -0.05, "Label = cross − in-domain macro-F1 (red: drop > 0.02).", ha="right", fontsize=6,
         color=MUTED)
plt.tight_layout(w_pad=3)
save(fig, "fig4_domain_shift")

# =============================================================================
# Fig 5  Headline comparison per test domain
# =============================================================================
cmp = [("Zero-shot\nNLI\n0 labels", "prompt-nli", lambda t: f1("prompt-nli", f"{t} (zero-shot)", 0)
        if False else R[(R.method == "prompt-nli") & (R.setting == f"{t} (zero-shot)")].f1.item()),
       ("NLI+LR\n\n64 labels", "prompt-nli", lambda t: f1("prompt-nli", f"{t}->{t}", 64)),
       ("BERT\n\n256 labels", "bert-head", lambda t: f1("bert-head", f"{t}->{t}", 256)),
       ("NLI+LR\n\nfull", "prompt-nli", lambda t: f1("prompt-nli", f"{t}->{t}", "full")),
       ("BERT\n\nfull", "bert-head", lambda t: f1("bert-head", f"{t}->{t}", "full"))]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3), sharey=True)
for ax, t, title in zip(axs, ["R", "L"], ["Restaurant (in-domain)", "Laptop (in-domain)"]):
    vals = [fn(t) for _, _, fn in cmp]
    cols = [C[m] for _, m, _ in cmp]
    alphas = [1, 0.75, 1, 0.75, 1]
    for i, (v, c) in enumerate(zip(vals, cols)):
        ax.bar(i, v, color=c, width=0.62, alpha=1 if "full" not in cmp[i][0] else 0.55,
               hatch=None if i != 0 else None, edgecolor="white", lw=1)
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.2, color=INK)
    ax.set_xticks(range(len(cmp))); ax.set_xticklabels([c[0] for c in cmp], fontsize=6.8)
    ax.set_title(title, fontsize=9); ax.grid(axis="x", visible=False); ax.set_ylim(0.5, 0.80)
axs[0].set_ylabel("macro-F1")
save(fig, "fig5_headline_comparison")

# =============================================================================
# Fig 6  Accuracy vs macro-F1: the majority-class gap
# =============================================================================
fig, ax = plt.subplots(figsize=(3.5, 3.0))
for m in ["prompt-cloze", "prompt-both", "prompt-nli", "bert-head"]:
    d = R[R.method == m]
    rest = d.setting.str.endswith("R") | d.setting.str.startswith("R (")
    ax.scatter(d.acc[rest], d.f1[rest], s=20, color=C[m], marker=MARK[m], edgecolor="white", lw=0.5)
    ax.scatter(d.acc[~rest], d.f1[~rest], s=20, facecolor="white", edgecolor=C[m], marker=MARK[m], lw=1.1)
lim = [0.5, 0.9]; ax.plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (3, 2)))
ax.text(0.735, 0.752, "acc = F1", fontsize=6.5, color=MUTED, rotation=47)
ax.set_xlim(0.6, 0.86); ax.set_ylim(0.48, 0.78)
ax.set_xlabel("accuracy"); ax.set_ylabel("macro-F1")
hs = [mpl.lines.Line2D([], [], marker=MARK[m], ls="", color=C[m], ms=5) for m in
      ["prompt-nli", "prompt-both", "prompt-cloze", "bert-head"]]
hs += [mpl.lines.Line2D([], [], marker="o", ls="", color=INK2, ms=5),
       mpl.lines.Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=5)]
ax.legend(hs, [NAME[m] for m in ["prompt-nli", "prompt-both", "prompt-cloze", "bert-head"]] +
          ["restaurant test (65% pos.)", "laptop test (53% pos.)"], fontsize=6, loc="upper left",
          handletextpad=0.2, labelspacing=0.3)
ax.text(0.845, 0.50, "gap grows with\nmajority-class share", fontsize=6.2, color=INK2, ha="right")
save(fig, "fig6_acc_vs_f1")

# =============================================================================
# Fig 7  Objective 2: implicit vs explicit aspect categories
# =============================================================================
grp = ["word absent (implicit)", "word present"]
meths = [("NLI zero-shot", "NLI zero-shot", "#2a78d6", 754),
         ("ATSC->ACSC (nli, terms-trained)", "NLI + LR (term-trained)", "#1baf7a", 754),
         ("NLI zero-shot (same 200)", "NLI zero-shot (200 subset)", "#86b6ef", 200),
         ("LLM direct", "Qwen2.5-1.5B direct", "#6250d6", 200),
         ("LLM chain-of-thought", "Qwen2.5-1.5B CoT", "#e87ba4", 200)]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=False)
for ax, metric, title in zip(axs, ["f1", "acc"], ["macro-F1", "accuracy"]):
    w = 0.15
    for i, (k, lab, col, n) in enumerate(meths):
        v = [O2[(O2.method == k) & (O2.group == g)][metric].item() for g in grp]
        xs = np.arange(2) + (i - 2) * w
        ax.bar(xs, v, width=w - 0.02, color=col, label=lab)
        for x, vv in zip(xs, v):
            ax.text(x, vv + 0.006, f"{vv:.2f}", ha="center", fontsize=5.6, color=INK2, rotation=90, va="bottom")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["category word absent\n(implicit proxy)", "category word present"])
    ax.set_title(title, fontsize=9); ax.grid(axis="x", visible=False)
    ax.set_ylim(0.55, 1.0) if metric == "f1" else ax.set_ylim(0.75, 1.02)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.08), fontsize=6.6, handlelength=1)
fig.text(0.995, -0.05, "Full set: 491 absent / 263 present. LLM subset: 100 / 100 (seed 1). "
         "Accuracy-F1 gap reflects class imbalance; macro-F1 is the primary metric.",
         ha="right", fontsize=6, color=MUTED)
plt.tight_layout(w_pad=2.5)
save(fig, "fig7_objective2_implicit")

# =============================================================================
# Fig 8  Chain-of-thought flip analysis (from cot_flips.txt)
# =============================================================================
txt = open("cot_flips.txt").read()
rec = re.findall(r"gold=(\w+) direct=(\w+) cot=(\w+) \| category word in text: (\w+)", txt)
idx = {c: i for i, c in enumerate(CLS)}
T = np.zeros((3, 3), int)
fixed = broken = both = 0
for g, d, c, _ in rec:
    T[idx[d], idx[c]] += 1
    if c == g: fixed += 1
    elif d == g: broken += 1
    else: both += 1
assert (fixed, broken, both) == (3, 8, 4), (fixed, broken, both)

# manual error taxonomy of the 15 flips (order = file order). See README for the per-example reading.
TAX = ["fixed", "pragmatic", "inconsistent", "no-opinion", "truncated", "pragmatic", "pragmatic", "inconsistent",
       "fixed", "comparative", "comparative", "comparative", "truncated", "fixed", "comparative"]
assert len(TAX) == len(rec)
TAX_L = {"fixed": "corrected by CoT", "comparative": "comparative / litotes /\ncounterfactual misread",
         "pragmatic": "literal reading of\npragmatic cue", "inconsistent": "reasoning contradicts\nfinal label",
         "truncated": "hit 120-token cap →\nparser default (neutral)", "no-opinion": "no opinion extracted\n→ neutral"}

fig = plt.figure(figsize=(7.6, 2.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.7, 1.2], wspace=0.95)
ax = fig.add_subplot(gs[0])
ax.imshow(T, cmap=mpl.colors.LinearSegmentedColormap.from_list("b", ["#ffffff", "#2a78d6"]), vmin=0,
          vmax=T.max())
for i in range(3):
    for j in range(3):
        ax.text(j, i, "–" if i == j else T[i, j], ha="center", va="center", fontsize=9,
                color="white" if T[i, j] > T.max() * 0.6 else INK)
ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(["pos", "neg", "neu"])
ax.set_yticklabels(["pos", "neg", "neu"]); ax.set_xlabel("CoT prediction"); ax.set_ylabel("direct prediction")
ax.grid(False); ax.set_title("(a) where the 15 flips go", fontsize=8.5)
for s in ax.spines.values(): s.set_visible(False)

ax = fig.add_subplot(gs[1])
v = [fixed, broken, both]; labs = ["fixed", "broken", "both\nwrong"]
cols = ["#1baf7a", "#e34948", "#b5b4ae"]
ax.bar(range(3), v, color=cols, width=0.6)
for i, x in enumerate(v): ax.text(i, x + 0.15, str(x), ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=7); ax.set_ylim(0, 9.5)
ax.grid(axis="x", visible=False); ax.set_ylabel("flips")
ax.set_title(f"(b) net effect {fixed - broken:+d} / 200", fontsize=8.5)

ax = fig.add_subplot(gs[2])
order = ["comparative", "pragmatic", "inconsistent", "truncated", "no-opinion", "fixed"]
cnt = [TAX.count(k) for k in order]
ax.barh(range(len(order))[::-1], cnt, color=["#e34948"] * 5 + ["#1baf7a"], height=0.6)
for j, c in zip(range(len(order))[::-1], cnt): ax.text(c + 0.08, j, str(c), va="center", fontsize=7.5)
ax.set_yticks(range(len(order))[::-1]); ax.set_yticklabels([TAX_L[k] for k in order], fontsize=6.4)
ax.set_xlim(0, 5); ax.grid(axis="y", visible=False); ax.set_xlabel("flips")
ax.set_title("(c) what changed the label", fontsize=8.5)
save(fig, "fig8_cot_flip_analysis")
print("CoT transition matrix (rows direct, cols CoT):\n", T)
print("flips to neutral:", T[:, 2].sum(), "| from neutral:", T[2, :].sum())
print("done ->", sorted(os.listdir(OUT)))
