# ============ EXTERNAL VALIDATION: SemEval-2016 Task 5 ============
# Needs from cell 1 (already run): D, F, LABELS, dev, DT, score(), AutoTokenizer,
# AutoModelForSequenceClassification, AutoModelForMaskedLM, run_bert (only if RUN_BASELINE was True).
import os, subprocess, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, torch
from tqdm.auto import tqdm
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM

# ---- 0. get SemEval-2016 Task 5 English restaurant/laptop XML from the public mirror ----
REPO = "/content/ABSADataset"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "--depth", "1",
                     "https://github.com/YaxinCui/ABSADataset.git", REPO], check=True)
BASE = f"{REPO}/SemEval2016Task5Corrected"

RESTAURANT_ENTITY_WORD = {
    "AMBIENCE": "ambience", "DRINKS": "drinks", "FOOD": "food",
    "LOCATION": "location", "RESTAURANT": "restaurant", "SERVICE": "service",
}
LAPTOP_ENTITY_WORD = {
    "BATTERY": "battery", "COMPANY": "company", "CPU": "processor",
    "DISPLAY": "screen", "FANS_COOLING": "fan", "GRAPHICS": "graphics card",
    "HARDWARE": "hardware", "HARD_DISC": "hard drive", "KEYBOARD": "keyboard",
    "LAPTOP": "laptop", "MEMORY": "memory", "MOTHERBOARD": "motherboard",
    "MOUSE": "mouse", "MULTIMEDIA_DEVICES": "multimedia", "OPTICAL_DRIVES": "optical drive",
    "OS": "operating system", "PORTS": "ports", "POWER_SUPPLY": "power supply",
    "SHIPPING": "shipping", "SOFTWARE": "software", "SUPPORT": "support", "WARRANTY": "warranty",
}

def parse_opinions(path):
    rows = []
    for sent in ET.parse(path).getroot().iter("sentence"):
        text_el = sent.find("text")
        if text_el is None or text_el.text is None:
            continue
        text = text_el.text
        ops = sent.find("Opinions")
        if ops is None:
            continue
        for op in ops.findall("Opinion"):
            polarity = op.get("polarity")
            if polarity not in LABELS:
                continue
            rows.append((text, op.get("category"), op.get("target"), polarity))
    return rows

def to_rows(parsed, entity_word_map, explicit_only=False, category_only=False):
    out = []
    for text, category, target, polarity in parsed:
        entity = category.split("#")[0]
        word = entity_word_map.get(entity, entity.lower().replace("_", " "))
        is_explicit = target not in (None, "NULL")
        if explicit_only and not is_explicit:
            continue
        aspect = word if category_only or not is_explicit else target
        out.append((text, aspect, LABELS.index(polarity)))
    return out

r_tr_raw = parse_opinions(f"{BASE}/train/restaurants_train_english.xml")
r_te_raw = parse_opinions(f"{BASE}/test/restaurants_test_english.xml")
l_tr_raw = parse_opinions(f"{BASE}/train/laptops_train_english.xml")
l_te_raw = parse_opinions(f"{BASE}/test/laptops_test_english.xml")

S = {
    "R16_tr": to_rows(r_tr_raw, RESTAURANT_ENTITY_WORD, explicit_only=True),
    "R16_te": to_rows(r_te_raw, RESTAURANT_ENTITY_WORD, explicit_only=True),
    "RC16_tr": to_rows(r_tr_raw, RESTAURANT_ENTITY_WORD, category_only=True),
    "RC16_te": to_rows(r_te_raw, RESTAURANT_ENTITY_WORD, category_only=True),
    "LC16_tr": to_rows(l_tr_raw, LAPTOP_ENTITY_WORD, category_only=True),
    "LC16_te": to_rows(l_te_raw, LAPTOP_ENTITY_WORD, category_only=True),
}
for k, v in S.items():
    print(k, len(v), "pos/neg/neu =", np.bincount([r[2] for r in v], minlength=3))

# ---- reload frozen scorers (deleted after cell 1's feature caching) ----
BS = 32
ntok = AutoTokenizer.from_pretrained("roberta-large-mnli")
nli = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli", torch_dtype=DT).to(dev).eval()
ENT = nli.config.label2id["ENTAILMENT"]
HYP = ["The {} is good.", "The {} is bad.", "The {} is okay."]

@torch.no_grad()
def nli_feats(rows):
    out = []
    for i in tqdm(range(0, len(rows), BS), desc="NLI (sem16)"):
        b, f = rows[i:i+BS], []
        for h in HYP:
            e = ntok([r[0] for r in b], [h.format(r[1]) for r in b], return_tensors="pt",
                     padding=True, truncation=True, max_length=256).to(dev)
            f.append(nli(**e).logits[:, ENT].float().cpu())
        out.append(torch.stack(f, 1))
    return log_softmax(torch.cat(out).numpy(), axis=1)

mtok = AutoTokenizer.from_pretrained("roberta-large")
mlm = AutoModelForMaskedLM.from_pretrained("roberta-large", torch_dtype=DT).to(dev).eval()
WORDS = [[" good", " great"], [" bad", " terrible"], [" okay", " fine"]]
IDS = [[mtok.encode(w, add_special_tokens=False)[0] for w in ws] for ws in WORDS]

@torch.no_grad()
def cloze_feats(rows):
    out = []
    for i in tqdm(range(0, len(rows), BS), desc="Cloze (sem16)"):
        b = rows[i:i+BS]
        e = mtok([f"{r[0]} The {r[1]} is <mask>." for r in b], return_tensors="pt", padding=True).to(dev)
        pos = (e.input_ids == mtok.mask_token_id).nonzero()
        lp = mlm(**e).logits[pos[:, 0], pos[:, 1]].float().log_softmax(-1)
        out.append(torch.stack([lp[:, g].logsumexp(-1) for g in IDS], 1).cpu())
    return log_softmax(torch.cat(out).numpy(), axis=1)

FS = {k: dict(nli=nli_feats(v), cloze=cloze_feats(v), y=np.array([r[2] for r in v])) for k, v in S.items()}
del nli, mlm; torch.cuda.empty_cache()

def Xf(d, mode):
    return {"nli": d["nli"], "cloze": d["cloze"], "both": np.hstack([d["nli"], d["cloze"]])}[mode]

sem16_results = []

# ---- 1. zero-shot on all three SemEval-2016 test sets ----
for te in ["R16_te", "RC16_te", "LC16_te"]:
    d = FS[te]
    for name, s in [("prompt-nli", d["nli"]), ("prompt-cloze", d["cloze"]), ("prompt-both", d["nli"] + d["cloze"])]:
        a, f = score(d["y"], s.argmax(1))
        sem16_results.append(dict(method=name, setting=f"{te} (zero-shot)", n=0, acc=a, f1=f))

# ---- 2. in-domain few-shot calibration (5 seeds), restaurant-term, restaurant-category, laptop-category ----
IN_DOMAIN = [("R16_tr", "R16_te", "R16->R16"), ("LC16_tr", "LC16_te", "LC16->LC16"),
             ("RC16_tr", "RC16_te", "RC16->RC16")]
for mode in ["nli", "cloze", "both"]:
    for tr, te, tag in IN_DOMAIN:
        Xtr, ytr, Xte, yte = Xf(FS[tr], mode), FS[tr]["y"], Xf(FS[te], mode), FS[te]["y"]
        for n in [16, 64, 256, 1024, "full"]:
            m = len(ytr) if n == "full" else min(n, len(ytr))
            r = []
            for s in range(1 if n == "full" else 5):
                idx = np.random.RandomState(s).permutation(len(ytr))[:m]
                clf = LogisticRegression(max_iter=3000).fit(Xtr[idx], ytr[idx])
                r.append(score(yte, clf.predict(Xte)))
            a, f = np.mean(r, 0)
            sem16_results.append(dict(method=f"prompt-{mode}", setting=tag, n=m, acc=a, f1=f))
print("In-domain 2016 few-shot done.")

# ---- 3. cross-dataset (2014 <-> 2016), restaurant term-level only ----
CROSS = [("R_tr", "R16_te", "F2014R->F2016R"), ("R16_tr", "R_te", "F2016R->F2014R")]
for mode in ["nli", "cloze", "both"]:
    for tr, te, tag in CROSS:
        Ftr = F[tr] if tr in F else FS[tr]
        Fte = F[te] if te in F else FS[te]
        Xtr, ytr, Xte, yte = Xf(Ftr, mode), Ftr["y"], Xf(Fte, mode), Fte["y"]
        for n in [64, 256, "full"]:
            m = len(ytr) if n == "full" else min(n, len(ytr))
            r = []
            for s in range(1 if n == "full" else 5):
                idx = np.random.RandomState(s).permutation(len(ytr))[:m]
                clf = LogisticRegression(max_iter=3000).fit(Xtr[idx], ytr[idx])
                r.append(score(yte, clf.predict(Xte)))
            a, f = np.mean(r, 0)
            sem16_results.append(dict(method=f"prompt-{mode}", setting=tag, n=m, acc=a, f1=f))
print("Cross-dataset 2014<->2016 done.")

# ---- 4. BERT baseline on 2016 restaurant term, in-domain + cross-dataset ----
if "run_bert" in dir():
    for n in [256, "full"]:
        rows = S["R16_tr"] if n == "full" else [S["R16_tr"][i] for i in np.random.RandomState(0).permutation(len(S["R16_tr"]))[:n]]
        print(f"BERT baseline: train R16_tr, n={len(rows)} ...")
        res = run_bert(rows, {"R16_te": S["R16_te"], "R_te": D["R_te"]}, epochs=3 if n == "full" else 10)
        for te, (a, f) in res.items():
            sem16_results.append(dict(method="bert-head", setting=f"R16->{te}", n=len(rows), acc=a, f1=f))
    for n in [256, "full"]:
        rows = D["R_tr"] if n == "full" else [D["R_tr"][i] for i in np.random.RandomState(0).permutation(len(D["R_tr"]))[:n]]
        print(f"BERT baseline: train R_tr (2014), n={len(rows)} ...")
        res = run_bert(rows, {"R_te": D["R_te"], "R16_te": S["R16_te"]}, epochs=3 if n == "full" else 10)
        for te, (a, f) in res.items():
            sem16_results.append(dict(method="bert-head", setting=f"R14->{te}", n=len(rows), acc=a, f1=f))
else:
    print("run_bert not in scope (RUN_BASELINE was False in cell 1) -- skipping BERT baseline.")

R16 = pd.DataFrame(sem16_results)
R16.to_csv("sem16_results.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== SemEval-2016 ZERO-SHOT ===")
print(R16[R16.setting.str.contains("zero-shot")].round(3).to_string(index=False))
print("\n=== SemEval-2016 IN-DOMAIN / CROSS-DATASET (macro-F1) ===")
print(R16[~R16.setting.str.contains("zero-shot")].pivot_table(index=["method", "n"], columns="setting", values="f1").round(3))

try:
    from google.colab import files
    files.download("sem16_results.csv")
except Exception as e:
    print("Results saved as sem16_results.csv in the Colab file panel.", e)
