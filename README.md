# Open Aspect Target Sentiment Classification via Prompting

Prompt-based Aspect Target Sentiment Classification (ATSC) with frozen pretrained
models (RoBERTa-large-MNLI for NLI-style prompting, RoBERTa-large MLM for cloze
prompting) vs. a fine-tuned BERT-base baseline. Evaluated on SemEval-2014 Task 4
(restaurant + laptop), with external validation on SemEval-2016 Task 5.

## Repo layout

| Path | What it is |
|---|---|
| `code/minor_project_with_metrics.ipynb` | Main notebook: zero-shot + few-shot NLI/cloze prompting, BERT baseline, metrics. |
| `code/cell_metrics_obj134.py`, `code/cell_metrics_obj2.py` | Standalone versions of the metrics cells (confusion matrices, calibration, per-category F1, significance tests). |
| `code/make_figures.py` | Regenerates Figures 1–8 from `results/atsc_results.csv`, `results/objective2_results.csv`, `results/cot_flips.txt`. |
| `code/make_figures_run_9_16.py` | Regenerates Figures 9–16 (confusion matrices, reliability, learning curves, per-class F1) from `results/metrics_table.json`, `results/seed_perclass.json`. |
| `code/build_datasets.py`, `code/cell_sem16_v2.py`, `code/make_fig17.py` | SemEval-2016 external-validation pipeline: clones the SemEval-2016 XML data, re-runs zero-shot/few-shot/cross-dataset comparisons, builds Figure 17. |
| `code/export.py` | Colab export helper. |
| `figs/` | All paper figures, PNG. |
| `results/` | Raw results: metrics CSVs, per-seed stats, CoT flip log, SemEval-2016 run output. |

## Figures

**Fig. 1 — Method overview.** A frozen RoBERTa-large-MNLI scores three hypotheses ("The {aspect} is good / bad / okay.") against the review. A frozen RoBERTa-large fills "{review} The {aspect} is \<mask\>." and its label-word probabilities are pooled per class. Each branch gives a 3-way log-softmax, used directly (zero-shot argmax) or fed to a logistic regression trained on n = 16 to all labelled examples.

**Fig. 2 — Label distribution.** SemEval-2014 v2 files, conflict removed. Restaurant test is 65% positive, which is why accuracy overstates performance there and macro-F1 is the primary metric.

**Fig. 3 — Label efficiency.** Macro-F1 against the number of labelled training examples (log scale) for the four train→test settings. Dashed lines are each branch's zero-shot score. Prompt-NLI with 64 labels matches or beats BERT trained on 256 in every setting.

**Fig. 4 — Robustness to domain shift.** Each row compares training on the test domain (filled) with training on the other domain (hollow), on the same test set. With full data the prompt models move by at most 0.010. BERT loses 0.096 on restaurant test (full) and 0.065–0.100 at n = 256.

**Fig. 5 — Headline comparison.** Zero-shot NLI with no labels comes within 0.011 of fully supervised BERT on restaurants (0.741 vs 0.752) and within 0.010 on laptops (0.691 vs 0.701). NLI + logistic regression on 64 labels beats full BERT on laptops (0.743 vs 0.701).

**Fig. 6 — Accuracy vs macro-F1, all runs.** Restaurant runs (filled) sit far below the diagonal. Accuracy is inflated by the positive majority, so macro-F1 is the metric that tells the systems apart.

**Fig. 7 — Objective 2: implicit vs explicit categories.** Macro-F1 and accuracy are shown separately for category queries whose category word is absent from or present in the review. Every method scores *higher* macro-F1 on the absent group — probably reflects differences in class mix between the groups rather than "implicit is easier."

**Fig. 8 — What chain-of-thought changed.** (a) Direct→CoT transitions over the 15 flipped predictions. 7 of the 15 flips go *to* neutral. (b) 3 fixed, 8 broken, 4 wrong either way — net −5 of 200, McNemar exact p = 0.23, not significant. (c) Manual error taxonomy of the flips (below).

**Figs. 9–11, 15 → `figs/confusion_matrices_all.png`.** All confusion matrices (restaurant, laptop, cross-domain transfer, and the Objective-2 implicit/explicit split) are combined into one image instead of four separate files, in run order: Restaurant → Laptop → Cross-domain → Objective 2.

**Fig. 12 — Reliability / calibration.** Confidence calibration curves + expected calibration error (ECE) per model.

**Fig. 13 — Learning curves with std bands.** Macro-F1 vs. n labelled examples, mean ± 1 std over 5 seeds, plus a class-balanced logistic-regression line.

**Fig. 14 — Per-class F1.** Per-class (pos/neg/neu) F1 broken out by model.

**Fig. 16 — Objective 2 per-category F1.** Per-category breakdown of the implicit-vs-explicit comparison in Fig. 7.

**Fig. 17 — SemEval-2016 external validation.** Zero-shot macro-F1 by prompt (NLI / Cloze / Both) across SemEval-2014 and SemEval-2016, restaurant and laptop, explicit and category-level. Headline: zero-shot NLI gets 0.739 macro-F1 on SemEval-2016 vs. 0.741 for fine-tuned BERT on SemEval-2014 — the zero-shot approach replicates almost exactly on a different year's data with a different annotation schema.

## Metrics and tables

`cm_restaurant`, `cm_laptop`, `cm_cross_domain`, `cm_objective2` (row-normalised confusion matrices with counts, now combined into one figure — see above); `reliability` (calibration + ECE); `learning_curves_std` (±1 std bands, plus a class-balanced LR line); `per_category_f1`; `results/metrics_table.json` (acc, macro-F1 + 95% bootstrap CI, weighted F1, balanced acc, Cohen's κ, MCC, per-class P/R/F1, ECE); `results/seed_perclass.json` (mean ± std over 5 draws); SemEval-2016 run output in `results/sem16_raw_output.txt`.

## Limitations

SemEval-2016 external validation covers explicit aspect targets for restaurant (~75% of opinions have an explicit OTE) but the laptop split is category-level only (0% explicit targets in that year's schema), so the laptop cross-year comparison is weaker than the restaurant one.

## CoT flip taxonomy (check before citing)

| # | Review (short) | gold / direct / CoT | Category |
|---|---|---|---|
| 1 | pub fare … portabella lasagna | neu / pos / neu | fixed |
| 2 | Tables are close … bumping elbows | neg / neu / pos | literal reading of pragmatic cue |
| 3 | narrow space … tasty treats, dirt-cheap | pos / pos / neg | reasoning contradicts label |
| 4 | Try the rose roll | pos / neg / neu | no opinion extracted |
| 5 | Black Sheep … sports-bar butch | neg / pos / neu | truncated (empty answer → neutral) |
| 6 | go when the place is quiet | neu / neu / neg | literal/pragmatic |
| 7 | Dug the blue bar area too | pos / pos / neu | literal/pragmatic ("too" misread) |
| 8 | Food was decent, but not great | pos / neu / neg | reasoning hedges then picks neg |
| 9 | so good … waiting a nightmare | pos / neg / pos | fixed |
| 10 | can make better at home | neg / neg / neu | comparative |
| 11 | prices would be 3x as high | pos / pos / neg | comparative |
| 12 | prices are not terrible | pos / pos / neu | litotes |
| 13 | better for 1/3 the price | neg / neg / neu | truncated at 120 tokens |
| 14 | jealous competing restaurants | pos / neg / pos | fixed |
| 15 | you'd expect it on par | neg / neg / pos | counterfactual |

Note: #8 is a questionable gold label. "Decent, but not great" annotated positive is arguable, so this flip may not be a true error.
