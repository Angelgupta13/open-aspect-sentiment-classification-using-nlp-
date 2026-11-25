# ============ EXTENDED METRICS (Objectives 1, 3, 4) — run AFTER cell 1 ============
# Needs from cell 1: D, F, LABELS, dev, DT, AutoTokenizer, AutoModelForSequenceClassification.
# Outputs (in /content/paper_metrics/): metrics_table.csv, fewshot_seed_stats.csv, significance.csv,
#   predictions.npz and figures: confusion matrices, reliability diagrams, learning curves with error bands.
RUN_BERT_PREDS = True        # re-trains BERT (full + 256, both domains, ~8 min) to get per-example predictions
N_BOOT = 1000

import os, numpy as np, pandas as pd, torch, matplotlib as mpl, matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
                             cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score)
OUTD = "/content/paper_metrics" if os.path.isdir("/content") else "paper_metrics"
os.makedirs(OUTD, exist_ok=True)
SHORT = ["pos", "neg", "neu"]
mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "savefig.bbox": "tight"})

# ---------------------------------------------------------------- metric helpers
def ece(y, P, bins=10):
    """Expected calibration error (top-label, equal-width bins)."""
    conf, pred = P.max(1), P.argmax(1); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e

def boot_ci(y, p, fn, n=N_BOOT, seed=0):
    rng = np.random.RandomState(seed); s = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y)); s.append(fn(y[i], p[i]))
    return np.percentile(s, [2.5, 97.5])

def mf1(y, p):
    """Macro-F1 over the 3 classes (same as sklearn average='macro', labels=[0,1,2], zero_division=0), fast."""
    cm = np.bincount(3 * np.asarray(y) + np.asarray(p), minlength=9).reshape(3, 3)
    tp = np.diag(cm); den = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))

def full_metrics(name, setting, y, p, P=None):
    pr, rc, f, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    lo, hi = boot_ci(y, p, mf1)
    r = dict(method=name, setting=setting, n_test=len(y),
             acc=accuracy_score(y, p), macro_f1=mf1(y, p), macro_f1_ci_lo=lo, macro_f1_ci_hi=hi,
             weighted_f1=f1_score(y, p, average="weighted", zero_division=0),
             balanced_acc=balanced_accuracy_score(y, p), kappa=cohen_kappa_score(y, p), mcc=matthews_corrcoef(y, p))
    for k, c in enumerate(SHORT):
        r[f"P_{c}"], r[f"R_{c}"], r[f"F1_{c}"] = pr[k], rc[k], f[k]
    r["ece"] = ece(y, P) if P is not None else np.nan
    return r

def mcnemar(y, pa, pb):
    """Exact McNemar on discordant pairs. b = A right & B wrong, c = A wrong & B right."""
    b = int(((pa == y) & (pb != y)).sum()); c = int(((pa != y) & (pb == y)).sum())
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)

def paired_boot_delta(y, pa, pb, n=N_BOOT, seed=0):
    rng = np.random.RandomState(seed); d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y)); d.append(mf1(y[i], pa[i]) - mf1(y[i], pb[i]))
    d = np.array(d); return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d <= 0).mean()

# ---------------------------------------------------------------- 1. prompt models: predictions + probs
PRED = {}      # key -> (y, pred, prob)
def Xf(d, mode): return {"nli": d["nli"], "both": np.hstack([d["nli"], d["cloze"]]), "cloze": d["cloze"]}[mode]
for te in ["R_te", "L_te"]:
    d = F[te]
    PRED[f"NLI zero-shot|{te[0]}"]   = (d["y"], d["nli"].argmax(1), softmax(d["nli"], 1))
    PRED[f"Cloze zero-shot|{te[0]}"] = (d["y"], d["cloze"].argmax(1), softmax(d["cloze"], 1))
    s = d["nli"] + d["cloze"]
    PRED[f"NLI+Cloze zero-shot|{te[0]}"] = (d["y"], s.argmax(1), softmax(s, 1))

SETTINGS = [("R_tr", "R_te", "R->R"), ("L_tr", "L_te", "L->L"), ("R_tr", "L_te", "R->L"), ("L_tr", "R_te", "L->R")]
seed_rows = []
for mode in ["nli", "cloze", "both"]:
    for cw in [None, "balanced"]:                       # 'balanced' = NEW ablation (class-weighted LR)
        for tr, te, tag in SETTINGS:
            Xtr, ytr, Xte, yte = Xf(F[tr], mode), F[tr]["y"], Xf(F[te], mode), F[te]["y"]
            for n in [16, 64, 256, 1024, "full"]:
                m = len(ytr) if n == "full" else min(n, len(ytr))
                f1s, accs = [], []
                for s in range(1 if n == "full" else 5):
                    idx = np.random.RandomState(s).permutation(len(ytr))[:m]
                    if len(np.unique(ytr[idx])) < 2: continue
                    clf = LogisticRegression(max_iter=3000, class_weight=cw).fit(Xtr[idx], ytr[idx])
                    p = clf.predict(Xte); f1s.append(mf1(yte, p)); accs.append(accuracy_score(yte, p))
                    if s == 0 and n in (64, "full"):
                        P = np.zeros((len(yte), 3)); P[:, clf.classes_] = clf.predict_proba(Xte)
                        PRED[f"{mode.upper()} LR{'-bal' if cw else ''} n={n}|{tag}"] = (yte, p, P)
                seed_rows.append(dict(method=f"prompt-{mode}", class_weight=cw or "none", setting=tag, n=m,
                                      f1_mean=np.mean(f1s), f1_std=np.std(f1s, ddof=1) if len(f1s) > 1 else 0,
                                      acc_mean=np.mean(accs), acc_std=np.std(accs, ddof=1) if len(accs) > 1 else 0,
                                      seeds=len(f1s)))
S = pd.DataFrame(seed_rows); S.to_csv(f"{OUTD}/fewshot_seed_stats.csv", index=False)
print("Few-shot mean ± std (macro-F1), unweighted LR, NLI:")
print(S[(S.method == "prompt-nli") & (S.class_weight == "none")]
      .assign(f1=lambda d: d.f1_mean.round(3).astype(str) + " ± " + d.f1_std.round(3).astype(str))
      .pivot_table(index="n", columns="setting", values="f1", aggfunc="first"))

# ---------------------------------------------------------------- 2. BERT with predictions + probs
if RUN_BERT_PREDS:
    btok = AutoTokenizer.from_pretrained("bert-base-uncased")
    def enc(rows):
        return btok([r[0] for r in rows], [r[1] for r in rows], truncation=True, max_length=128,
                    padding=True, return_tensors="pt")
    def run_bert_preds(train_rows, test_sets, epochs, seed=0, bs=32, lr=2e-5):
        torch.manual_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr); rng = np.random.RandomState(seed)
        for _ in range(epochs):
            model.train(); order = rng.permutation(len(train_rows))
            for i in range(0, len(order), bs):
                b = [train_rows[j] for j in order[i:i+bs]]
                e = {k: v.to(dev) for k, v in enc(b).items()}
                model(**e, labels=torch.tensor([r[2] for r in b]).to(dev)).loss.backward()
                opt.step(); opt.zero_grad()
        model.eval(); out = {}
        for name, rows in test_sets.items():
            logits = []
            with torch.no_grad():
                for i in range(0, len(rows), 64):
                    e = {k: v.to(dev) for k, v in enc(rows[i:i+64]).items()}
                    logits.append(model(**e).logits.float().cpu().numpy())
            L = np.concatenate(logits); out[name] = (np.array([r[2] for r in rows]), L.argmax(1), softmax(L, 1))
        del model; torch.cuda.empty_cache(); return out
    for trk, t_in, t_x in [("R_tr", "R_te", "L_te"), ("L_tr", "L_te", "R_te")]:
        for n in [256, "full"]:
            rows = D[trk] if n == "full" else [D[trk][i] for i in np.random.RandomState(0).permutation(len(D[trk]))[:n]]
            print(f"BERT (with predictions): train {trk}, n={len(rows)}")
            res = run_bert_preds(rows, {t_in: D[t_in], t_x: D[t_x]}, epochs=3 if n == "full" else 10)
            for te, v in res.items():
                PRED[f"BERT n={n}|{trk[0]}->{te[0]}"] = v

# ---------------------------------------------------------------- 3. metrics table
rows = []
for k, (y, p, P) in PRED.items():
    name, setting = k.split("|"); rows.append(full_metrics(name, setting, y, p, P))
M = pd.DataFrame(rows); M.to_csv(f"{OUTD}/metrics_table.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
show = ["method", "setting", "acc", "macro_f1", "macro_f1_ci_lo", "macro_f1_ci_hi", "kappa", "mcc", "F1_pos",
        "F1_neg", "F1_neu", "ece"]
print("\n=== EXTENDED METRICS (95% bootstrap CI on macro-F1) ===")
print(M[show].round(3).to_string(index=False))
np.savez_compressed(f"{OUTD}/predictions.npz", **{k.replace("|", "__").replace(">", "").replace(" ", "_"):
                    np.stack([v[0], v[1]]) for k, v in PRED.items()})

# ---------------------------------------------------------------- 4. significance tests
pairs = [("NLI zero-shot|R", "BERT n=full|R->R"), ("NLI zero-shot|L", "BERT n=full|L->L"),
         ("NLI LR n=64|R->R", "BERT n=256|R->R"), ("NLI LR n=64|L->L", "BERT n=256|L->L"),
         ("NLI LR n=full|R->L", "BERT n=full|R->L"), ("NLI LR n=full|L->R", "BERT n=full|L->R"),
         ("NLI zero-shot|R", "NLI LR n=full|R->R"), ("NLI LR n=full|R->R", "NLI LR-bal n=full|R->R")]
sig = []
for a, b in pairs:
    if a not in PRED or b not in PRED: continue
    y, pa, _ = PRED[a]; _, pb, _ = PRED[b]
    bb, cc, pv = mcnemar(y, pa, pb); dm, dlo, dhi, p_boot = paired_boot_delta(y, pa, pb)
    sig.append(dict(A=a, B=b, A_right_B_wrong=bb, A_wrong_B_right=cc, mcnemar_p=pv,
                    dF1_A_minus_B=dm, dF1_ci_lo=dlo, dF1_ci_hi=dhi, p_boot_A_not_better=p_boot))
SIG = pd.DataFrame(sig); SIG.to_csv(f"{OUTD}/significance.csv", index=False)
print("\n=== SIGNIFICANCE (McNemar on accuracy; paired bootstrap on macro-F1) ===")
print(SIG.round(4).to_string(index=False))

# ---------------------------------------------------------------- 5. figures
def plot_cms(keys, titles, fname, suptitle=None):
    fig, axs = plt.subplots(1, len(keys), figsize=(1.85 * len(keys) + 0.4, 2.2))
    axs = np.atleast_1d(axs)
    for ax, k, t in zip(axs, keys, titles):
        if k not in PRED: ax.axis("off"); continue
        y, p, _ = PRED[k]; cm = confusion_matrix(y, p, labels=[0, 1, 2]); cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmn[i, j]*100:.0f}%\n({cm[i, j]})", ha="center", va="center", fontsize=6.5,
                        color="white" if cmn[i, j] > 0.55 else "#111")
        ax.set_xticks(range(3)); ax.set_xticklabels(SHORT); ax.set_yticks(range(3)); ax.set_yticklabels(SHORT)
        ax.set_title(f"{t}\nmacro-F1 {mf1(y, p):.3f}", fontsize=8); ax.set_xlabel("predicted", fontsize=7.5)
        for s in ax.spines.values(): s.set_visible(False)
    for ax in axs[1:]: ax.set_yticklabels([])
    axs[0].set_ylabel("gold", fontsize=7.5); fig.subplots_adjust(wspace=0.12)
    if suptitle: fig.suptitle(suptitle, fontsize=9, y=1.08)
    fig.savefig(f"{OUTD}/{fname}.png", dpi=300); fig.savefig(f"{OUTD}/{fname}.pdf"); plt.show()

plot_cms(["NLI zero-shot|R", "NLI LR n=64|R->R", "NLI LR n=full|R->R", "BERT n=256|R->R", "BERT n=full|R->R"],
         ["NLI zero-shot", "NLI+LR (64)", "NLI+LR (full)", "BERT (256)", "BERT (full)"],
         "cm_restaurant", "Restaurant test — row-normalised confusion matrices")
plot_cms(["NLI zero-shot|L", "NLI LR n=64|L->L", "NLI LR n=full|L->L", "BERT n=256|L->L", "BERT n=full|L->L"],
         ["NLI zero-shot", "NLI+LR (64)", "NLI+LR (full)", "BERT (256)", "BERT (full)"],
         "cm_laptop", "Laptop test — row-normalised confusion matrices")
plot_cms(["NLI LR n=full|R->L", "BERT n=full|R->L", "NLI LR n=full|L->R", "BERT n=full|L->R"],
         ["NLI+LR R→L", "BERT R→L", "NLI+LR L→R", "BERT L→R"], "cm_cross_domain", "Cross-domain (full source data)")

# reliability diagrams
def reliability(keys, titles, fname, bins=10):
    fig, axs = plt.subplots(1, len(keys), figsize=(2.0 * len(keys), 2.2), sharey=True)
    for ax, k, t in zip(np.atleast_1d(axs), keys, titles):
        if k not in PRED: ax.axis("off"); continue
        y, p, P = PRED[k]; conf = P.max(1); edges = np.linspace(0, 1, bins + 1); xs, accs, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5: xs.append(conf[m].mean()); accs.append((p[m] == y[m]).mean()); ns.append(m.sum())
        ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=0.8)
        ax.plot(xs, accs, marker="o", ms=3.5, color="#2a78d6", lw=1.6)
        ax.set_title(f"{t}\nECE {ece(y, P):.3f}", fontsize=8); ax.set_xlabel("confidence", fontsize=7.5)
        ax.set_xlim(1/3, 1); ax.set_ylim(0, 1.02); ax.grid(color="#eee", lw=0.6)
    np.atleast_1d(axs)[0].set_ylabel("accuracy", fontsize=7.5)
    fig.savefig(f"{OUTD}/{fname}.png", dpi=300); fig.savefig(f"{OUTD}/{fname}.pdf"); plt.show()
reliability(["NLI zero-shot|R", "NLI LR n=full|R->R", "BERT n=full|R->R",
             "NLI zero-shot|L", "NLI LR n=full|L->L", "BERT n=full|L->L"],
            ["NLI 0-shot (R)", "NLI+LR (R)", "BERT (R)", "NLI 0-shot (L)", "NLI+LR (L)", "BERT (L)"], "reliability")

# learning curves with ±1 std bands (now that std is saved)
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
COL = {"prompt-nli": "#2a78d6", "prompt-both": "#1baf7a", "prompt-cloze": "#eb6834"}
for ax, st in zip(axs, ["R->R", "L->L", "R->L", "L->R"]):
    for m, c in COL.items():
        d = S[(S.method == m) & (S.setting == st) & (S.class_weight == "none")].sort_values("n")
        ax.plot(d.n, d.f1_mean, color=c, marker="o", ms=3, lw=1.5, label=m.replace("prompt-", ""))
        ax.fill_between(d.n, d.f1_mean - d.f1_std, d.f1_mean + d.f1_std, color=c, alpha=0.18, lw=0)
        db = S[(S.method == m) & (S.setting == st) & (S.class_weight == "balanced")].sort_values("n")
        if m == "prompt-nli": ax.plot(db.n, db.f1_mean, color=c, ls=":", lw=1.3, label="nli, class-balanced LR")
    ax.set_xscale("log"); ax.set_title(st, fontsize=8.5); ax.grid(color="#eee", lw=0.6); ax.minorticks_off()
    ax.set_xticks([16, 64, 256, 1024]); ax.set_xticklabels(["16", "64", "256", "1k"], fontsize=7)
axs[0].set_ylabel("macro-F1"); axs[0].legend(fontsize=6, frameon=False)
fig.supxlabel("labelled examples (log); band = ±1 std over 5 draws", fontsize=8, y=-0.06)
fig.savefig(f"{OUTD}/learning_curves_std.png", dpi=300); fig.savefig(f"{OUTD}/learning_curves_std.pdf"); plt.show()

# ---------------------------------------------------------------- 6. software versions for the paper
import sklearn, transformers, scipy
print("\nVersions -> torch", torch.__version__, "| transformers", transformers.__version__,
      "| scikit-learn", sklearn.__version__, "| scipy", scipy.__version__,
      "| GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

try:
    import shutil; from google.colab import files
    shutil.make_archive("/content/paper_metrics", "zip", OUTD); files.download("/content/paper_metrics.zip")
except Exception as e:
    print("Saved to", OUTD, e)
