# Prints all metric outputs as one JSON blob so they can be copied out of the notebook.
import json, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
out = {}
out["metrics_table"] = M.to_dict(orient="records")
out["significance"] = SIG.to_dict(orient="records")
out["seed_stats"] = S.to_dict(orient="records")
out["cm"] = {k: confusion_matrix(v[0], v[1], labels=[0, 1, 2]).tolist() for k, v in PRED.items()}
rel = {}
for k, (y, p, P) in PRED.items():
    conf = P.max(1); edges = np.linspace(0, 1, 11); b = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum(): b.append([float(conf[m].mean()), float((p[m] == y[m]).mean()), int(m.sum())])
    rel[k] = b
out["reliability"] = rel
try:
    out["obj2_per_category"] = PC.to_dict(orient="records")
    out["obj2_sig"] = sig
    out["obj2_counts"] = {"all": np.bincount(yc, minlength=3).tolist(),
                          "absent": np.bincount(yc[absent], minlength=3).tolist(),
                          "present": np.bincount(yc[~absent], minlength=3).tolist(),
                          "sub_absent": np.bincount(ys[ab], minlength=3).tolist(),
                          "sub_present": np.bincount(ys[~ab], minlength=3).tolist()}
    out["obj2_cm"] = {t: confusion_matrix(y, p, labels=[0, 1, 2]).tolist() for y, p, t in panels}
except NameError as e:
    out["obj2_error"] = str(e)
import sklearn, transformers, torch
out["versions"] = dict(torch=torch.__version__, transformers=transformers.__version__, sklearn=sklearn.__version__,
                       gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
s = json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o))
open("/content/all_metrics.json", "w").write(s)
print("JSON_LEN", len(s))
