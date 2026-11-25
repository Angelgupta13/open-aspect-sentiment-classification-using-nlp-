"""Figures 9-16 from the extended-metrics GPU run (results.json, seed_perclass.json)."""
import json, os, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
OUT = "figs"; os.makedirs(OUT, exist_ok=True)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
CLS_C = ["#1baf7a", "#e34948", "#8a8984"]; SH = ["pos", "neg", "neu"]
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04})
def save(fig, n): fig.savefig(f"{OUT}/{n}.png", dpi=300); fig.savefig(f"{OUT}/{n}.pdf"); plt.close(fig)
D = json.load(open("results.json")); P = json.load(open("seed_perclass.json"))
BLUES = mpl.colors.LinearSegmentedColormap.from_list("b", ["#ffffff", "#2a78d6"])

def mf1(cm):
    cm = np.array(cm); tp = np.diag(cm); den = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0)))

def cm_grid(panels, name, ncols=3, sup=None):
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.35 * nrows))
    axs = np.atleast_1d(axs).ravel()
    for ax, (cm, title) in zip(axs, panels):
        cm = np.array(cm); n = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(n, cmap=BLUES, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{n[i, j]*100:.0f}%\n({cm[i, j]})", ha="center", va="center", fontsize=6.6,
                        color="white" if n[i, j] > 0.6 else INK, linespacing=1.1)
        ax.set_xticks(range(3)); ax.set_xticklabels(SH, fontsize=7.5); ax.set_yticks(range(3))
        ax.set_yticklabels(SH, fontsize=7.5); ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(f"{title}\nmacro-F1 {mf1(cm):.3f}", fontsize=7.8)
    for ax in axs[len(panels):]: ax.axis("off")
    for k, ax in enumerate(axs[:len(panels)]):
        if k % ncols == 0: ax.set_ylabel("gold", fontsize=7.5)
        if k >= len(panels) - ncols: ax.set_xlabel("predicted", fontsize=7.5)
    fig.subplots_adjust(wspace=0.25, hspace=0.55)
    save(fig, name)

C = D["cm"]
# Fig 9 / 10: in-domain confusion matrices
for dom, tag, name in [("R", "R->R", "fig9_cm_restaurant"), ("L", "L->L", "fig10_cm_laptop")]:
    cm_grid([(C[f"NLI zero-shot|{dom}"], "NLI zero-shot"), (C[f"NLI LR n=64|{tag}"], "NLI+LR, 64 labels"),
             (C[f"NLI LR n=full|{tag}"], "NLI+LR, full"), (C[f"NLI LR-bal n=full|{tag}"], "NLI+LR balanced, full"),
             (C[f"BERT n=256|{tag}"], "BERT, 256 labels"), (C[f"BERT n=full|{tag}"], "BERT, full")], name)
# Fig 11: cross-domain
cm_grid([(C["NLI LR n=full|R->L"], "NLI+LR  R→L"), (C["NLI LR-bal n=full|R->L"], "NLI+LR bal.  R→L"),
         (C["BERT n=full|R->L"], "BERT  R→L"), (C["NLI LR n=full|L->R"], "NLI+LR  L→R"),
         (C["NLI LR-bal n=full|L->R"], "NLI+LR bal.  L→R"), (C["BERT n=full|L->R"], "BERT  L→R")],
        "fig11_cm_cross_domain")

# Fig 12: reliability diagrams
keys = [("NLI zero-shot|R", "NLI zero-shot (Rest.)", "#2a78d6"), ("NLI LR n=full|R->R", "NLI+LR full (Rest.)", "#1baf7a"),
        ("BERT n=full|R->R", "BERT full (Rest.)", "#3d3c39"), ("NLI zero-shot|L", "NLI zero-shot (Lap.)", "#2a78d6"),
        ("NLI LR n=full|L->L", "NLI+LR full (Lap.)", "#1baf7a"), ("BERT n=full|L->L", "BERT full (Lap.)", "#3d3c39")]
fig, axs = plt.subplots(1, 6, figsize=(8.4, 2.0), sharey=True)
for ax, (k, t, col) in zip(axs, keys):
    b = np.array(D["rel"][k]); b = b[b[:, 2] >= 5]
    ax.plot([1/3, 1], [1/3, 1], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.plot(b[:, 0], b[:, 1], color=col, marker="o", ms=3, lw=1.5)
    ax.set_xlim(0.33, 1); ax.set_ylim(0.3, 1.02); ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xticklabels(["0.5", ".75", "1"], fontsize=7)
    ax.set_title(f"{t}\nECE {D['ece'][k]:.3f}", fontsize=6.6)
axs[0].set_ylabel("accuracy"); fig.supxlabel("confidence (bins with ≥ 5 examples)", fontsize=8, y=-0.1)
save(fig, "fig12_reliability")

# Fig 13: learning curves with +-1 std and class-balanced variant
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=True)
COL = {"nli": "#2a78d6", "both": "#1baf7a", "cloze": "#eb6834"}
NM = {"nli": "Prompt-NLI", "both": "Prompt-NLI+Cloze", "cloze": "Prompt-Cloze"}
for ax, st in zip(axs, ["R->R", "L->L", "R->L", "L->R"]):
    for m, cw, s, ns, f, sd in P["S"]:
        if s != st: continue
        ns, f, sd = np.array(ns), np.array(f), np.array(sd)
        if cw == "none":
            ax.plot(ns, f, color=COL[m], marker="o", ms=3, lw=1.5, label=NM[m])
            ax.fill_between(ns, f - sd, f + sd, color=COL[m], alpha=0.15, lw=0)
        elif m in ("nli", "both"):
            ax.plot(ns, f, color=COL[m], lw=1.4, ls=(0, (1.2, 1.2)), label=f"{NM[m]}, class-balanced")
    bn, bf = P["BERT"][st]
    ax.plot(bn, bf, color="#3d3c39", marker="D", ms=4, lw=1, ls=(0, (1, 1.5)), label="BERT fine-tuned")
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xticks([16, 64, 256, 1024, ns[-1]]); ax.set_xticklabels(["16", "64", "256", "1k", "full"], fontsize=7)
    ax.set_title(st.replace("->", " → ").replace("R", "Rest.").replace("L", "Lap."), fontsize=8.5)
axs[0].set_ylabel("macro-F1"); axs[0].set_ylim(0.4, 0.82)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=6.8)
fig.supxlabel("labelled training examples (log); shaded band = ±1 std over 5 random subsets", fontsize=7.5, y=-0.06)
save(fig, "fig13_learning_curves_std")

# Fig 14: per-class F1 (where the errors are)
PC = P["perclass"]; cols = P["perclass_cols"]
rows = [("NLI zero-shot|R", "NLI\n0-shot"), ("NLI LR n=full|R->R", "NLI+LR"), ("NLI LR-bal n=full|R->R", "NLI+LR\nbalanced"),
        ("BOTH LR-bal n=full|R->R", "NLI+Cloze\nbalanced"), ("BERT n=full|R->R", "BERT")]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
for ax, dom, title in zip(axs, ["R", "L"], ["Restaurant test (in-domain, full data)", "Laptop test (in-domain, full data)"]):
    rr = [(k.replace("R->R", f"{dom}->{dom}").replace("|R", f"|{dom}") if dom == "L" else k, lab) for k, lab in rows]
    w = 0.26
    for c, (cn, colr) in enumerate(zip(["F1_pos", "F1_neg", "F1_neu"], CLS_C)):
        v = [PC[k][cols.index(cn)] for k, _ in rr]
        xs = np.arange(len(rr)) + (c - 1) * w
        ax.bar(xs, v, width=w - 0.03, color=colr, label=["positive", "negative", "neutral"][c])
        for x, vv in zip(xs, v): ax.text(x, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=5.4, color=INK2, rotation=90)
    ax.set_xticks(range(len(rr))); ax.set_xticklabels([l for _, l in rr], fontsize=7); ax.set_title(title, fontsize=8.5)
    ax.grid(axis="x", visible=False); ax.set_ylim(0, 1.08)
axs[0].set_ylabel("per-class F1"); axs[0].legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(1.1, 1.25))
save(fig, "fig14_per_class_f1")

# Fig 15: Objective 2 confusion matrices
O = D["obj2_cm"]
cm_grid([(O["NLI 0-shot, word absent"], "NLI 0-shot, word absent\n(n=491)"),
         (O["NLI 0-shot, word present"], "NLI 0-shot, word present\n(n=263)"),
         (O["NLI 0-shot (200)"], "NLI 0-shot, 200 subset"), (O["Qwen direct (200)"], "Qwen direct, 200 subset"),
         (O["Qwen CoT (200)"], "Qwen CoT, 200 subset")], "fig15_cm_objective2", ncols=5)

# Fig 16: Objective 2 per-category macro-F1 and neutral support
pc = [r for r in D["PC"] if r[1] != "all"]; cats = ["food", "service", "ambience", "price"]
fig, ax = plt.subplots(figsize=(5.4, 2.4)); w = 0.36
for i, (m, col) in enumerate([("NLI zero-shot", "#2a78d6"), ("NLI+LR (term-trained)", "#1baf7a")]):
    v = [next(r[5] for r in pc if r[0] == m and r[1] == c) for c in cats]
    xs = np.arange(4) + (i - 0.5) * w
    ax.bar(xs, v, width=w - 0.03, color=col, label=m)
    for x, vv in zip(xs, v): ax.text(x, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=6.3, color=INK2)
lab = []
for c in cats:
    r = next(r for r in pc if r[0] == "NLI zero-shot" and r[1] == c)
    lab.append(f"{c}\nn={r[2]}, {r[3]} implicit\n{r[10]} neutral")
ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=6.5); ax.set_ylim(0, 1.0); ax.set_ylabel("macro-F1")
ax.grid(axis="x", visible=False); ax.legend(fontsize=6.8, loc="upper right", ncol=1)
save(fig, "fig16_objective2_per_category")
print(sorted(os.listdir(OUT)))
