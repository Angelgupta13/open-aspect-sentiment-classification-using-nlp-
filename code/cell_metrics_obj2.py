# ============ EXTENDED METRICS (Objective 2) — run AFTER the Objective 2 cell ============
# Needs from the Objective 2 cell: C_te, yc, absent, Fc, Ftr, sel, ys, ab, p_dir, p_cot, LABELS.
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             cohen_kappa_score)
OUTD = "/content/paper_metrics_obj2" if os.path.isdir("/content") else "paper_metrics_obj2"
os.makedirs(OUTD, exist_ok=True)
SHORT = ["pos", "neg", "neu"]
def mf1(y, p):
    """Macro-F1 over the 3 classes (same as sklearn average='macro', labels=[0,1,2], zero_division=0), fast."""
    cm = np.bincount(3 * np.asarray(y) + np.asarray(p), minlength=9).reshape(3, 3)
    tp = np.diag(cm); den = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))

cats = np.array([r[1] for r in C_te])
p_nli = Fc["nli"].argmax(1)
clf = LogisticRegression(max_iter=3000).fit(Ftr["nli"], Ftr["y"]); p_lr = clf.predict(Fc["nli"])

# class balance of the category queries (explains the accuracy / macro-F1 gap)
print("Category-query label distribution (pos/neg/neu):", np.bincount(yc, minlength=3),
      "| absent:", np.bincount(yc[absent], minlength=3), "| present:", np.bincount(yc[~absent], minlength=3))

# per-category and per-class results
rows = []
for name, p in [("NLI zero-shot", p_nli), ("NLI+LR (term-trained)", p_lr)]:
    for c in ["all"] + sorted(set(cats)):
        m = np.ones(len(yc), bool) if c == "all" else cats == c
        pr, rc, f, sup = precision_recall_fscore_support(yc[m], p[m], labels=[0, 1, 2], zero_division=0)
        rows.append(dict(method=name, category=c, n=int(m.sum()), n_absent=int((m & absent).sum()),
                         acc=accuracy_score(yc[m], p[m]), macro_f1=mf1(yc[m], p[m]), kappa=cohen_kappa_score(yc[m], p[m]),
                         F1_pos=f[0], F1_neg=f[1], F1_neu=f[2], support_neu=int(sup[2])))
PC = pd.DataFrame(rows); PC.to_csv(f"{OUTD}/per_category.csv", index=False)
print(PC.round(3).to_string(index=False))

# significance: LLM direct vs CoT, NLI vs LLM direct (same 200)
def mcnemar(y, pa, pb):
    b = int(((pa == y) & (pb != y)).sum()); c = int(((pa != y) & (pb == y)).sum())
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)
nli_sub = Fc["nli"][sel].argmax(1)
sig = []
for a, pa, b, pb in [("LLM direct", p_dir, "LLM CoT", p_cot), ("NLI zero-shot", nli_sub, "LLM direct", p_dir)]:
    x, z, pv = mcnemar(ys, pa, pb)
    sig.append(dict(A=a, B=b, A_right_B_wrong=x, A_wrong_B_right=z, mcnemar_p=round(pv, 4)))
print(pd.DataFrame(sig).to_string(index=False)); pd.DataFrame(sig).to_csv(f"{OUTD}/significance.csv", index=False)

# confusion matrices
def cm_panel(ax, y, p, title):
    cm = confusion_matrix(y, p, labels=[0, 1, 2]); cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]*100:.0f}%\n({cm[i, j]})", ha="center", va="center", fontsize=6.5,
                    color="white" if cmn[i, j] > 0.55 else "#111")
    ax.set_xticks(range(3)); ax.set_xticklabels(SHORT); ax.set_yticks(range(3)); ax.set_yticklabels(SHORT)
    ax.set_title(f"{title}\nmacro-F1 {mf1(y, p):.3f}", fontsize=7.5); ax.set_xlabel("predicted", fontsize=7)
    for s in ax.spines.values(): s.set_visible(False)
panels = [(yc[absent], p_nli[absent], "NLI 0-shot, word absent"), (yc[~absent], p_nli[~absent], "NLI 0-shot, word present"),
          (ys, nli_sub, "NLI 0-shot (200)"), (ys, p_dir, "Qwen direct (200)"), (ys, p_cot, "Qwen CoT (200)")]
fig, axs = plt.subplots(1, len(panels), figsize=(10, 2.3))
for ax, (y, p, t) in zip(axs, panels): cm_panel(ax, y, p, t)
for ax in axs[1:]: ax.set_yticklabels([])
axs[0].set_ylabel("gold", fontsize=7); fig.subplots_adjust(wspace=0.12)
fig.savefig(f"{OUTD}/cm_objective2.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{OUTD}/cm_objective2.pdf", bbox_inches="tight"); plt.show()

# per-category macro-F1 bar
fig, ax = plt.subplots(figsize=(4.2, 2.2))
d = PC[PC.category != "all"]; cs = sorted(set(d.category)); w = 0.38
for i, (name, col) in enumerate([("NLI zero-shot", "#2a78d6"), ("NLI+LR (term-trained)", "#1baf7a")]):
    v = [d[(d.method == name) & (d.category == c)].macro_f1.item() for c in cs]
    ax.bar(np.arange(len(cs)) + (i - 0.5) * w, v, width=w - 0.03, color=col, label=name)
ns = [d[(d.method == "NLI zero-shot") & (d.category == c)] for c in cs]
ax.set_xticks(range(len(cs))); ax.set_xticklabels([f"{c}\n(n={r.n.item()}, {r.n_absent.item()} implicit)" for c, r in zip(cs, ns)], fontsize=6.5)
ax.set_ylabel("macro-F1"); ax.legend(fontsize=6.5, frameon=False); ax.set_ylim(0, 1)
fig.savefig(f"{OUTD}/per_category_f1.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{OUTD}/per_category_f1.pdf", bbox_inches="tight"); plt.show()

try:
    import shutil; from google.colab import files
    shutil.make_archive("/content/paper_metrics_obj2", "zip", OUTD); files.download("/content/paper_metrics_obj2.zip")
except Exception as e:
    print("Saved to", OUTD, e)
