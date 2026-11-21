# Final results tables

Generated from the CSVs under `results/` (nothing re-trained). Branch `akshay`; `git pull --ff-only` reported "Already up to date" before this file was written. **Nothing was committed.**

## Missing inputs (not guessed)

- `Harshith's stress error analysis` — error analysis, stress part (no file in results/, docs/ or on any remote branch mentions it)
- No cross-dataset (BoL→RLT, RLT→BoL) deception **fine-tuning** results and no within-RLT deception fine-tuning results exist (`results/deception/finetune*.csv` only hold train=bol/test=bol and train=bol+rlt/test=bol+rlt). Those cells are shown as *not run*.
- No **pooled** and no **cross-dataset** deception **from-scratch** results exist (`results/deception/scratch_video.csv` only has within_bol and within_rlt) and no pooled deception baseline (`baseline.csv` has within and cross only). Shown as *not run*.
- Own recordings: no such data in this project, so that table is intentionally left out.

## How to read the numbers

- **mean ± std** is over runs (a run = one test fold × one seed), sample std (ddof=1), as written by the scripts. Fold and seed variance are mixed in that std; runs are not independent. Values are shown to 3 decimals, p-values to 4 significant digits; full precision is in the source CSVs.
- **Runs / seeds differ by family. Always check the `runs` and `seeds` columns:**
  - pooled-feature baselines (stress and deception): 5 folds, **no seed** (deterministic logistic regression, `random_state` fixed at 42); the cross-dataset deception baseline is a **single** train/test split, so no std.
  - frozen-feature fusion (`deep`): 5 folds × 3 seeds (0,1,2) = 15 runs; deception cross-dataset: one split × 3 seeds = 3 runs.
  - **fine-tuning (LoRA): seed 0 only, 5 folds = 5 runs** — for deception *and* for stress (`results/stress/finetune.csv` also only has seed 0).
  - from-scratch deception video CNN: 5 folds × 3 seeds = 15 runs.
  - integration models (two-head, stress-aware, transfer): 5 folds × 3 seeds = 15 runs.
  - lighting: **one fold (fold 0) × 3 seeds**, 65 BoL test clips.
- **Chance marking (deception only).** `p` = two-sided binomial test of accuracy vs 0.5. For the frozen / scratch families it is copied from `deep.csv` / `scratch_video.csv` (k = correct clips averaged over seeds, n = test clips) and `p_worst` is the largest per-seed p. For fine-tuning, integration, leakage and lighting rows, which have no p in the CSVs, **I recomputed p from the `*_predictions.csv` files with the same rule as `src/deception/report.py`** (lighting: k = accuracy × 65 per seed). Rows with p > 0.05 are marked **NOT sig. above chance**. Clips of one person are not independent, so the p-values are optimistic, and none is corrected for the many comparisons in this file.

## 1. Stress by mode (MAS, 5 group-wise folds)

| family | mode | runs | seeds | accuracy | macro-F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| baseline (pooled LR) | audio | 5 folds | fixed 42, no repeats | 0.620 ± 0.070 | 0.592 ± 0.073 | 0.628 ± 0.107 |
| baseline (pooled LR) | video | 5 folds | fixed 42, no repeats | 0.492 ± 0.065 | 0.480 ± 0.071 | 0.494 ± 0.094 |
| baseline (pooled LR) | both | 5 folds | fixed 42, no repeats | 0.592 ± 0.051 | 0.561 ± 0.063 | 0.617 ± 0.107 |
| frozen features (deep) | audio | 15 | 0,1,2 | 0.595 ± 0.077 | 0.559 ± 0.069 | 0.614 ± 0.102 |
| frozen features (deep) | video | 15 | 0,1,2 | 0.494 ± 0.085 | 0.459 ± 0.075 | 0.490 ± 0.112 |
| frozen features (deep) | concat | 15 | 0,1,2 | 0.581 ± 0.095 | 0.538 ± 0.093 | 0.600 ± 0.106 |
| frozen features (deep) | late | 15 | 0,1,2 | 0.576 ± 0.091 | 0.537 ± 0.096 | 0.609 ± 0.102 |
| frozen features (deep) | xattn | 15 | 0,1,2 | 0.574 ± 0.091 | 0.533 ± 0.096 | 0.584 ± 0.120 |
| fine-tuning (LoRA r=8) | concat | 5 | 0 only | 0.517 ± 0.056 | 0.469 ± 0.045 | 0.478 ± 0.082 |
| fine-tuning (LoRA r=8) | audio | 5 | 0 only | 0.588 ± 0.112 | 0.499 ± 0.056 | 0.574 ± 0.139 |
| fine-tuning (LoRA r=8) | video | 5 | 0 only | 0.457 ± 0.087 | 0.444 ± 0.097 | 0.441 ± 0.088 |
| from scratch (audio CNN) | audio | 15 | 0,1,2 | 0.598 ± 0.104 | 0.561 ± 0.102 | 0.605 ± 0.129 |

Reference (same 5 folds, `baseline.csv`): always predicting the overall majority class (label 0) gives accuracy 0.511 ± 0.179 and macro-F1 0.331 ± 0.078; the training-fold majority class gives accuracy 0.411 ± 0.149 and macro-F1 0.285 ± 0.073. Fold test sizes are 79/72/54/132/48 clips, so the fold std is large.

Not in the table: `stress/finetune_concat_f0_s0*.csv` is a single extra re-run of fine-tuning concat on fold 0, seed 0 (accuracy 0.570, macro-F1 0.431, ROC-AUC 0.437); the same cell in `finetune.csv` gives 0.557 / 0.424 / 0.437, i.e. two runs with the same seed did not reproduce exactly.

## 2. Deception: within-dataset, pooled and cross-dataset, by mode and family

Setting names: `within_bol` / `within_rlt` = train and test inside one dataset; `pooled` = train on BoL+RLT folds, test reported separately for each dataset; `bol→rlt`, `rlt→bol` = train on one dataset, test on the other.

| family | setting | test set | mode | runs | seeds | accuracy | macro-F1 | ROC-AUC | p | p_worst | vs chance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (pooled LR) | within_bol | bol | audio | 5 | fixed 42, no repeats | 0.532 ± 0.076 | 0.518 ± 0.085 | 0.538 ± 0.067 | 0.2672 | — | NOT sig. above chance |
| baseline (pooled LR) | within_bol | bol | video | 5 | fixed 42, no repeats | 0.511 ± 0.048 | 0.496 ± 0.037 | 0.501 ± 0.025 | 0.7393 | — | NOT sig. above chance |
| baseline (pooled LR) | within_bol | bol | av | 5 | fixed 42, no repeats | 0.541 ± 0.034 | 0.523 ± 0.049 | 0.535 ± 0.058 | 0.1491 | — | NOT sig. above chance |
| baseline (pooled LR) | within_rlt | rlt | audio | 5 | fixed 42, no repeats | 0.518 ± 0.129 | 0.487 ± 0.105 | 0.513 ± 0.175 | 0.8558 | — | NOT sig. above chance |
| baseline (pooled LR) | within_rlt | rlt | video | 5 | fixed 42, no repeats | 0.572 ± 0.105 | 0.544 ± 0.100 | 0.610 ± 0.080 | 0.2029 | — | NOT sig. above chance |
| baseline (pooled LR) | within_rlt | rlt | av | 5 | fixed 42, no repeats | 0.541 ± 0.129 | 0.490 ± 0.138 | 0.601 ± 0.111 | 0.5856 | — | NOT sig. above chance |
| baseline (pooled LR) | bol_to_rlt | rlt | audio | 1 | fixed 42, no repeats | 0.554 (no std: 1 run) | 0.542 (no std: 1 run) | 0.558 (no std: 1 run) | 0.2753 | — | NOT sig. above chance |
| baseline (pooled LR) | bol_to_rlt | rlt | video | 1 | fixed 42, no repeats | 0.430 (no std: 1 run) | 0.374 (no std: 1 run) | 0.339 (no std: 1 run) | 0.1455 | — | NOT sig. above chance |
| baseline (pooled LR) | bol_to_rlt | rlt | av | 1 | fixed 42, no repeats | 0.455 (no std: 1 run) | 0.446 (no std: 1 run) | 0.432 (no std: 1 run) | 0.3634 | — | NOT sig. above chance |
| baseline (pooled LR) | rlt_to_bol | bol | audio | 1 | fixed 42, no repeats | 0.498 (no std: 1 run) | 0.333 (no std: 1 run) | 0.444 (no std: 1 run) | 1 | — | NOT sig. above chance |
| baseline (pooled LR) | rlt_to_bol | bol | video | 1 | fixed 42, no repeats | 0.514 (no std: 1 run) | 0.420 (no std: 1 run) | 0.518 (no std: 1 run) | 0.6573 | — | NOT sig. above chance |
| baseline (pooled LR) | rlt_to_bol | bol | av | 1 | fixed 42, no repeats | 0.471 (no std: 1 run) | 0.379 (no std: 1 run) | 0.455 (no std: 1 run) | 0.3181 | — | NOT sig. above chance |
| baseline (pooled LR) | pooled | bol / rlt | — | — | — | *not run* | *not run* | *not run* | — | — | — |
| frozen features (deep) | within_bol | bol | audio | 15 | 3 | 0.560 ± 0.067 | 0.547 ± 0.066 | 0.579 ± 0.060 | 0.03488 | 0.183 | p<0.05 (not robust: worst-seed p>0.05) |
| frozen features (deep) | within_bol | bol | video | 15 | 3 | 0.534 ± 0.033 | 0.526 ± 0.036 | 0.542 ± 0.037 | 0.2672 | 0.5057 | NOT sig. above chance |
| frozen features (deep) | within_bol | bol | concat | 15 | 3 | 0.538 ± 0.049 | 0.522 ± 0.061 | 0.573 ± 0.060 | 0.183 | 1 | NOT sig. above chance |
| frozen features (deep) | within_bol | bol | late | 15 | 3 | 0.548 ± 0.049 | 0.526 ± 0.052 | 0.575 ± 0.068 | 0.09594 | 0.3181 | NOT sig. above chance |
| frozen features (deep) | within_bol | bol | xattn | 15 | 3 | 0.557 ± 0.064 | 0.543 ± 0.070 | 0.593 ± 0.076 | 0.04567 | 0.5792 | p<0.05 (not robust: worst-seed p>0.05) |
| frozen features (deep) | within_rlt | rlt | audio | 15 | 3 | 0.551 ± 0.124 | 0.521 ± 0.134 | 0.604 ± 0.170 | 0.3634 | 0.8558 | NOT sig. above chance |
| frozen features (deep) | within_rlt | rlt | video | 15 | 3 | 0.595 ± 0.121 | 0.568 ± 0.126 | 0.653 ± 0.089 | 0.04505 | 0.4672 | p<0.05 (not robust: worst-seed p>0.05) |
| frozen features (deep) | within_rlt | rlt | concat | 15 | 3 | 0.670 ± 0.093 | 0.646 ± 0.099 | 0.701 ± 0.097 | 0.0002442 | 0.000983 | p<0.05 |
| frozen features (deep) | within_rlt | rlt | late | 15 | 3 | 0.702 ± 0.067 | 0.677 ± 0.095 | 0.710 ± 0.081 | 9.815e-06 | 2.316e-05 | p<0.05 |
| frozen features (deep) | within_rlt | rlt | xattn | 15 | 3 | 0.648 ± 0.095 | 0.621 ± 0.109 | 0.682 ± 0.110 | 0.001873 | 0.0287 | p<0.05 |
| frozen features (deep) | pooled | bol | audio | 15 | 3 | 0.546 ± 0.059 | 0.514 ± 0.070 | 0.566 ± 0.083 | 0.1203 | 0.4375 | NOT sig. above chance |
| frozen features (deep) | pooled | bol | video | 15 | 3 | 0.530 ± 0.068 | 0.517 ± 0.068 | 0.547 ± 0.071 | 0.3181 | 0.6573 | NOT sig. above chance |
| frozen features (deep) | pooled | bol | concat | 15 | 3 | 0.541 ± 0.066 | 0.523 ± 0.067 | 0.546 ± 0.086 | 0.1491 | 0.5057 | NOT sig. above chance |
| frozen features (deep) | pooled | bol | late | 15 | 3 | 0.516 ± 0.058 | 0.490 ± 0.070 | 0.554 ± 0.088 | 0.5792 | 0.9117 | NOT sig. above chance |
| frozen features (deep) | pooled | bol | xattn | 15 | 3 | 0.516 ± 0.060 | 0.488 ± 0.073 | 0.565 ± 0.085 | 0.5792 | 0.9117 | NOT sig. above chance |
| frozen features (deep) | pooled | rlt | audio | 15 | 3 | 0.514 ± 0.141 | 0.497 ± 0.145 | 0.544 ± 0.177 | 0.8558 | 1 | NOT sig. above chance |
| frozen features (deep) | pooled | rlt | video | 15 | 3 | 0.592 ± 0.129 | 0.561 ± 0.139 | 0.649 ± 0.163 | 0.04505 | 0.2029 | p<0.05 (not robust: worst-seed p>0.05) |
| frozen features (deep) | pooled | rlt | concat | 15 | 3 | 0.627 ± 0.142 | 0.615 ± 0.143 | 0.658 ± 0.165 | 0.006151 | 0.4672 | p<0.05 (not robust: worst-seed p>0.05) |
| frozen features (deep) | pooled | rlt | late | 15 | 3 | 0.577 ± 0.160 | 0.556 ± 0.153 | 0.630 ± 0.193 | 0.1455 | 0.8558 | NOT sig. above chance |
| frozen features (deep) | pooled | rlt | xattn | 15 | 3 | 0.619 ± 0.164 | 0.595 ± 0.171 | 0.656 ± 0.176 | 0.01061 | 0.5856 | p<0.05 (not robust: worst-seed p>0.05) |
| frozen features (deep) | rlt_to_bol | bol | audio | 3 | 3 | 0.512 ± 0.023 | 0.389 ± 0.075 | 0.480 ± 0.062 | 0.7393 | 1 | NOT sig. above chance |
| frozen features (deep) | rlt_to_bol | bol | video | 3 | 3 | 0.514 ± 0.035 | 0.466 ± 0.024 | 0.511 ± 0.073 | 0.6573 | 0.3748 | NOT sig. above chance |
| frozen features (deep) | rlt_to_bol | bol | concat | 3 | 3 | 0.483 ± 0.040 | 0.414 ± 0.054 | 0.494 ± 0.059 | 0.5792 | 1 | NOT sig. above chance |
| frozen features (deep) | rlt_to_bol | bol | late | 3 | 3 | 0.483 ± 0.054 | 0.453 ± 0.064 | 0.482 ± 0.076 | 0.5792 | 0.8244 | NOT sig. above chance |
| frozen features (deep) | rlt_to_bol | bol | xattn | 3 | 3 | 0.472 ± 0.075 | 0.429 ± 0.114 | 0.481 ± 0.098 | 0.3181 | 0.04567 | NOT sig. above chance |
| frozen features (deep) | bol_to_rlt | rlt | audio | 3 | 3 | 0.477 ± 0.029 | 0.338 ± 0.004 | 0.417 ± 0.134 | 0.7163 | 1 | NOT sig. above chance |
| frozen features (deep) | bol_to_rlt | rlt | video | 3 | 3 | 0.499 ± 0.033 | 0.466 ± 0.033 | 0.562 ± 0.097 | 1 | 1 | NOT sig. above chance |
| frozen features (deep) | bol_to_rlt | rlt | concat | 3 | 3 | 0.496 ± 0.014 | 0.341 ± 0.008 | 0.472 ± 0.052 | 1 | 1 | NOT sig. above chance |
| frozen features (deep) | bol_to_rlt | rlt | late | 3 | 3 | 0.496 ± 0.008 | 0.370 ± 0.049 | 0.553 ± 0.034 | 1 | 1 | NOT sig. above chance |
| frozen features (deep) | bol_to_rlt | rlt | xattn | 3 | 3 | 0.490 ± 0.010 | 0.403 ± 0.057 | 0.491 ± 0.052 | 0.8558 | 1 | NOT sig. above chance |
| fine-tuning (LoRA r=8) | within_bol | bol | audio | 5 | 0 only | 0.578 ± 0.020 | 0.566 ± 0.022 | 0.574 ± 0.034 | 0.005464 | 0.005464 | p<0.05 |
| fine-tuning (LoRA r=8) | within_bol | bol | video | 5 | 0 only | 0.565 ± 0.073 | 0.546 ± 0.092 | 0.585 ± 0.062 | 0.01968 | 0.01968 | p<0.05 |
| fine-tuning (LoRA r=8) | within_bol | bol | xattn | 5 | 0 only | 0.569 ± 0.036 | 0.548 ± 0.052 | 0.603 ± 0.039 | 0.01453 | 0.01453 | p<0.05 |
| fine-tuning (LoRA r=8) | pooled | bol | audio | 5 | 0 only | 0.566 ± 0.035 | 0.552 ± 0.043 | 0.598 ± 0.065 | 0.01968 | 0.01968 | p<0.05 |
| fine-tuning (LoRA r=8) | pooled | bol | video | 5 | 0 only | 0.563 ± 0.070 | 0.538 ± 0.062 | 0.550 ± 0.077 | 0.02635 | 0.02635 | p<0.05 |
| fine-tuning (LoRA r=8) | pooled | bol | xattn | 5 | 0 only | 0.532 ± 0.055 | 0.479 ± 0.096 | 0.560 ± 0.105 | 0.2672 | 0.2672 | NOT sig. above chance |
| fine-tuning (LoRA r=8) | pooled | rlt | audio | 5 | 0 only | 0.611 ± 0.143 | 0.597 ± 0.136 | 0.546 ± 0.146 | 0.01061 | 0.01061 | p<0.05 |
| fine-tuning (LoRA r=8) | pooled | rlt | video | 5 | 0 only | 0.547 ± 0.129 | 0.536 ± 0.124 | 0.596 ± 0.140 | 0.4672 | 0.4672 | NOT sig. above chance |
| fine-tuning (LoRA r=8) | pooled | rlt | xattn | 5 | 0 only | 0.663 ± 0.050 | 0.648 ± 0.059 | 0.724 ± 0.129 | 0.0004986 | 0.0004986 | p<0.05 |
| fine-tuning (LoRA r=8) | within_rlt | rlt | — | — | — | *not run* | *not run* | *not run* | — | — | — |
| fine-tuning (LoRA r=8) | bol→rlt, rlt→bol | rlt, bol | — | — | — | *not run* | *not run* | *not run* | — | — | — |
| from scratch (video CNN+GRU) | within_bol | bol | scratch_video | 15 | 3 | 0.544 ± 0.032 | 0.493 ± 0.082 | 0.580 ± 0.060 | 0.1203 | 0.2672 | NOT sig. above chance |
| from scratch (video CNN+GRU) | within_rlt | rlt | scratch_video | 15 | 3 | 0.501 ± 0.124 | 0.413 ± 0.150 | 0.528 ± 0.117 | 1 | 0.7163 | NOT sig. above chance |
| from scratch (video CNN+GRU) | pooled, bol→rlt, rlt→bol | — | — | — | — | *not run* | *not run* | *not run* | — | — | — |

Notes for this table:

- Baseline rows: `runs` = number of folds (cross-dataset: 1 split, so no std). `p` is on all out-of-fold predictions together; `p_worst` does not apply (no seeds).
- Frozen-feature cross-dataset rows (`rlt_to_bol`, `bol_to_rlt`) are only **3 runs** (1 split × 3 seeds); their std is over seeds only.
- **Fine-tuning used seed 0 only (5 runs = 5 folds).** `p_worst` equals `p` there (one seed). The per-dataset rows for the pooled fine-tuning model are recomputed by me from `finetune_predictions.csv` (per fold, then mean ± std over the 5 folds); `finetune_summary.csv` only reports the two datasets combined, copied below.
- Rows where `p` ≤ 0.05 but `p_worst` > 0.05 are not robust across seeds (the significance depends on which seed is used).

Fine-tuning, combined test sets exactly as in `results/deception/finetune_summary.csv` (seed 0 only, 5 runs):

| setting | mode | runs | seeds | accuracy | macro-F1 | ROC-AUC | p (recomputed) | vs chance |
|---|---|---|---|---|---|---|---|---|
| within_bol (train bol, test bol) | xattn | 5 | 0 only | 0.569 ± 0.036 | 0.548 ± 0.052 | 0.603 ± 0.039 | 0.01453 (n=325) | p<0.05 |
| within_bol (train bol, test bol) | audio | 5 | 0 only | 0.578 ± 0.020 | 0.566 ± 0.022 | 0.574 ± 0.034 | 0.005464 (n=325) | p<0.05 |
| within_bol (train bol, test bol) | video | 5 | 0 only | 0.565 ± 0.073 | 0.546 ± 0.092 | 0.585 ± 0.062 | 0.01968 (n=325) | p<0.05 |
| pooled (train bol+rlt, test bol+rlt combined) | xattn | 5 | 0 only | 0.567 ± 0.031 | 0.548 ± 0.049 | 0.615 ± 0.073 | 0.005153 (n=446) | p<0.05 |
| pooled (train bol+rlt, test bol+rlt combined) | audio | 5 | 0 only | 0.580 ± 0.039 | 0.572 ± 0.044 | 0.599 ± 0.078 | 0.0007558 (n=446) | p<0.05 |
| pooled (train bol+rlt, test bol+rlt combined) | video | 5 | 0 only | 0.556 ± 0.033 | 0.546 ± 0.035 | 0.560 ± 0.039 | 0.02023 (n=446) | p<0.05 |

## 3. Two-head (stress + deception) versus single-task

Two-head = `integration/multitask*.csv` (one model, stress head + deception head, trained on MAS+BoL+RLT, `concat` fusion). Single-task = the `concat` frozen-feature runs of the same 15 folds × seeds (`stress/deep.csv` for MAS; `deception/deep.csv` pooled rows for BoL and RLT; `deception/runs/pooled_summary.csv` for BoL+RLT combined). 15 runs, seeds 0,1,2 for both.

| test set | head | model | runs | seeds | accuracy | macro-F1 | ROC-AUC | binomial p (deception) — vs chance |
|---|---|---|---|---|---|---|---|---|
| mas | stress | single-task (concat) | 15 | 0,1,2 | 0.581 ± 0.095 | 0.538 ± 0.093 | 0.600 ± 0.106 | — |
| mas | stress | two-head (multitask) | 15 | 0,1,2 | 0.573 ± 0.094 | 0.547 ± 0.101 | 0.572 ± 0.134 | — |
| bol | deception | single-task (concat) | 15 | 0,1,2 | 0.541 ± 0.066 | 0.523 ± 0.067 | 0.546 ± 0.086 | 0.1491 — NOT sig. above chance |
| bol | deception | two-head (multitask) | 15 | 0,1,2 | 0.540 ± 0.053 | 0.523 ± 0.062 | 0.557 ± 0.076 | 0.1491 — NOT sig. above chance |
| rlt | deception | single-task (concat) | 15 | 0,1,2 | 0.627 ± 0.142 | 0.615 ± 0.143 | 0.658 ± 0.165 | 0.006151 — p<0.05 |
| rlt | deception | two-head (multitask) | 15 | 0,1,2 | 0.568 ± 0.178 | 0.541 ± 0.186 | 0.598 ± 0.164 | 0.2029 — NOT sig. above chance |
| bol+rlt | deception | single-task (concat) | 15 | 0,1,2 | 0.564 ± 0.058 | 0.554 ± 0.061 | 0.589 ± 0.078 | 0.006888 — p<0.05 |
| bol+rlt | deception | two-head (multitask) | 15 | 0,1,2 | 0.546 ± 0.060 | 0.537 ± 0.059 | 0.562 ± 0.071 | 0.06467 — NOT sig. above chance |

Paired difference, two-head − single-task, same fold and seed (`integration/multitask_vs_single.csv`):

| test set | head | metric | pairs | seeds | two-head mean | single mean | diff (mean ± std) | better / worse |
|---|---|---|---|---|---|---|---|---|
| mas | stress | accuracy | 15 | 0,1,2 | 0.573 | 0.581 | -0.008 ± 0.135 | 8 / 6 |
| mas | stress | macro_f1 | 15 | 0,1,2 | 0.547 | 0.538 | +0.009 ± 0.111 | 8 / 7 |
| mas | stress | roc_auc | 15 | 0,1,2 | 0.572 | 0.600 | -0.029 ± 0.098 | 5 / 10 |
| bol | deception | accuracy | 15 | 0,1,2 | 0.540 | 0.541 | -0.000 ± 0.054 | 5 / 8 |
| bol | deception | macro_f1 | 15 | 0,1,2 | 0.523 | 0.523 | +0.001 ± 0.066 | 8 / 7 |
| bol | deception | roc_auc | 15 | 0,1,2 | 0.557 | 0.546 | +0.011 ± 0.079 | 7 / 8 |
| rlt | deception | accuracy | 15 | 0,1,2 | 0.568 | 0.627 | -0.060 ± 0.190 | 5 / 9 |
| rlt | deception | macro_f1 | 15 | 0,1,2 | 0.541 | 0.615 | -0.074 ± 0.192 | 5 / 10 |
| rlt | deception | roc_auc | 15 | 0,1,2 | 0.598 | 0.658 | -0.060 ± 0.159 | 7 / 8 |
| bol+rlt | deception | accuracy | 15 | 0,1,2 | 0.546 | 0.564 | -0.018 ± 0.061 | 7 / 8 |
| bol+rlt | deception | macro_f1 | 15 | 0,1,2 | 0.537 | 0.554 | -0.017 ± 0.059 | 6 / 9 |
| bol+rlt | deception | roc_auc | 15 | 0,1,2 | 0.562 | 0.589 | -0.027 ± 0.098 | 5 / 10 |

In every row the std of the paired difference is larger than the absolute mean difference; the files contain no significance test for these differences.

## 4. Stress-aware versus plain two-head, and transfer versus from scratch (paired differences)

Stress-aware = two-head model whose deception head also receives the (detached) stress probability. Plain = the two-head model above. Transfer = stage 1 stress on MAS, stage 2 deception on BoL+RLT (`concat`); its from-scratch counterpart is the single-task pooled `concat` model. All: 5 folds × seeds 0,1,2 = 15 paired runs.

Absolute results of each model (15 runs; binomial p recomputed from the predictions for deception rows):

| model | test set | runs | seeds | accuracy | macro-F1 | ROC-AUC | binomial p — vs chance |
|---|---|---|---|---|---|---|---|
| stress-aware two-head | mas | 15 | 0,1,2 | 0.556 ± 0.076 | 0.525 ± 0.084 | 0.565 ± 0.127 | — (stress head) |
| stress-aware two-head | bol | 15 | 0,1,2 | 0.536 ± 0.053 | 0.511 ± 0.062 | 0.546 ± 0.090 | 0.2223 — NOT sig. above chance |
| stress-aware two-head | rlt | 15 | 0,1,2 | 0.632 ± 0.146 | 0.607 ± 0.148 | 0.665 ± 0.141 | 0.006151 — p<0.05 |
| stress-aware two-head | bol+rlt | 15 | 0,1,2 | 0.561 ± 0.058 | 0.547 ± 0.060 | 0.584 ± 0.067 | 0.012 — p<0.05 |
| transfer (MAS→BoL+RLT) | bol | 15 | 0,1,2 | 0.532 ± 0.045 | 0.513 ± 0.043 | 0.557 ± 0.082 | 0.2672 — NOT sig. above chance |
| transfer (MAS→BoL+RLT) | rlt | 15 | 0,1,2 | 0.588 ± 0.131 | 0.572 ± 0.133 | 0.639 ± 0.151 | 0.06859 — NOT sig. above chance |
| transfer (MAS→BoL+RLT) | bol+rlt | 15 | 0,1,2 | 0.547 ± 0.054 | 0.534 ± 0.053 | 0.580 ± 0.071 | 0.05209 — NOT sig. above chance |

**Stress-aware − plain two-head** (`integration/stress_aware_vs_plain.csv`; `mas` rows are the stress head, the rest the deception head):

| test set | metric | pairs | seeds | plain mean | stress-aware mean | diff (mean ± std) | better / worse / equal |
|---|---|---|---|---|---|---|---|
| mas | accuracy | 15 | 0,1,2 | 0.573 | 0.556 | -0.017 ± 0.074 | 4 / 6 / 5 |
| mas | macro_f1 | 15 | 0,1,2 | 0.547 | 0.525 | -0.022 ± 0.067 | 4 / 6 / 5 |
| mas | roc_auc | 15 | 0,1,2 | 0.572 | 0.565 | -0.007 ± 0.059 | 8 / 7 / 0 |
| bol | accuracy | 15 | 0,1,2 | 0.540 | 0.536 | -0.005 ± 0.046 | 6 / 5 / 4 |
| bol | macro_f1 | 15 | 0,1,2 | 0.523 | 0.511 | -0.012 ± 0.048 | 6 / 6 / 3 |
| bol | roc_auc | 15 | 0,1,2 | 0.557 | 0.546 | -0.011 ± 0.061 | 5 / 9 / 1 |
| rlt | accuracy | 15 | 0,1,2 | 0.568 | 0.632 | +0.064 ± 0.136 | 6 / 1 / 8 |
| rlt | macro_f1 | 15 | 0,1,2 | 0.541 | 0.607 | +0.066 ± 0.129 | 6 / 1 / 8 |
| rlt | roc_auc | 15 | 0,1,2 | 0.598 | 0.665 | +0.067 ± 0.086 | 11 / 1 / 3 |
| bol+rlt | accuracy | 15 | 0,1,2 | 0.546 | 0.561 | +0.015 ± 0.049 | 6 / 5 / 4 |
| bol+rlt | macro_f1 | 15 | 0,1,2 | 0.537 | 0.547 | +0.010 ± 0.052 | 6 / 6 / 3 |
| bol+rlt | roc_auc | 15 | 0,1,2 | 0.562 | 0.584 | +0.022 ± 0.067 | 10 / 5 / 0 |

**Transfer − from scratch** (`integration/transfer_vs_scratch.csv`; deception head, BoL+RLT training):

| test set | metric | pairs | seeds | from-scratch mean | transfer mean | diff (mean ± std) | better / worse / equal |
|---|---|---|---|---|---|---|---|
| bol | accuracy | 15 | 0,1,2 | 0.541 | 0.532 | -0.009 ± 0.045 | 6 / 6 / 3 |
| bol | macro_f1 | 15 | 0,1,2 | 0.523 | 0.513 | -0.010 ± 0.056 | 6 / 8 / 1 |
| bol | roc_auc | 15 | 0,1,2 | 0.546 | 0.557 | +0.011 ± 0.049 | 9 / 6 / 0 |
| rlt | accuracy | 15 | 0,1,2 | 0.627 | 0.588 | -0.039 ± 0.124 | 4 / 9 / 2 |
| rlt | macro_f1 | 15 | 0,1,2 | 0.615 | 0.572 | -0.043 ± 0.125 | 5 / 10 / 0 |
| rlt | roc_auc | 15 | 0,1,2 | 0.658 | 0.639 | -0.020 ± 0.093 | 5 / 9 / 1 |
| bol+rlt | accuracy | 15 | 0,1,2 | 0.564 | 0.547 | -0.017 ± 0.043 | 4 / 8 / 3 |
| bol+rlt | macro_f1 | 15 | 0,1,2 | 0.554 | 0.534 | -0.020 ± 0.054 | 6 / 9 / 0 |
| bol+rlt | roc_auc | 15 | 0,1,2 | 0.589 | 0.580 | -0.009 ± 0.040 | 6 / 8 / 1 |

In every row of both paired tables the std of the paired difference exceeds the absolute mean difference; the files contain no significance test.

## 5. False positives by predicted stress, and stress–deception correlation

Source: `integration/false_positives_by_stress.csv` (+ `_runs.csv`, `_summary.txt`). Stress comes from one fixed MAS-trained audio model (`stress/stress_scores_on_deception.csv`); **bands are relative**: high/low = above/below each dataset's median over truthful clips (bol 0.057, rlt 0.751). FPR = false positives among truthful test predictions (a clip appears once per run, 15 runs = 5 folds × seeds 0,1,2). `pooled FPR` = FP / predictions over all runs; `run mean ± std` is the mean ± std of the 15 per-run FPRs. Model: `plain` = plain two-head, `cond` = stress-conditioned two-head.

| dataset | model | band | truthful clips | groups | predictions | FP | pooled FPR | runs | run mean ± std |
|---|---|---|---|---|---|---|---|---|---|
| bol | plain | low | 82 | 24 | 246 | 121 | 0.492 | 15 | 0.507 ± 0.277 |
| bol | plain | high | 81 | 27 | 243 | 133 | 0.547 | 15 | 0.533 ± 0.183 |
| bol | plain | all | 163 | 34 | 489 | 254 | 0.519 | 15 | 0.517 ± 0.211 |
| bol | cond | low | 82 | 24 | 246 | 92 | 0.374 | 15 | 0.385 ± 0.302 |
| bol | cond | high | 81 | 27 | 243 | 119 | 0.490 | 15 | 0.482 ± 0.253 |
| bol | cond | all | 163 | 34 | 489 | 211 | 0.431 | 15 | 0.428 ± 0.259 |
| rlt | plain | low | 30 | 24 | 90 | 29 | 0.322 | 15 | 0.296 ± 0.243 |
| rlt | plain | high | 30 | 15 | 90 | 28 | 0.311 | 15 | 0.291 ± 0.223 |
| rlt | plain | all | 60 | 37 | 180 | 57 | 0.317 | 15 | 0.317 ± 0.161 |
| rlt | cond | low | 30 | 24 | 90 | 31 | 0.344 | 15 | 0.324 ± 0.209 |
| rlt | cond | high | 30 | 15 | 90 | 21 | 0.233 | 15 | 0.226 ± 0.259 |
| rlt | cond | all | 60 | 37 | 180 | 52 | 0.289 | 15 | 0.289 ± 0.196 |
| bol+rlt | plain | low | 112 | 48 | 336 | 150 | 0.446 | 15 | 0.457 ± 0.206 |
| bol+rlt | plain | high | 111 | 42 | 333 | 161 | 0.483 | 15 | 0.477 ± 0.119 |
| bol+rlt | plain | all | 223 | 71 | 669 | 311 | 0.465 | 15 | 0.463 ± 0.149 |
| bol+rlt | cond | low | 112 | 48 | 336 | 123 | 0.366 | 15 | 0.372 ± 0.224 |
| bol+rlt | cond | high | 111 | 42 | 333 | 140 | 0.420 | 15 | 0.425 ± 0.190 |
| bol+rlt | cond | all | 223 | 71 | 669 | 263 | 0.393 | 15 | 0.391 ± 0.187 |

High-stress minus low-stress FPR (pooled over runs; 95% group bootstrap CI, 5000 resamples by person/video):

| dataset | model | groups | high − low | 95% CI | CI vs 0 |
|---|---|---|---|---|---|
| bol | plain | 34 | +0.055 | [-0.117, +0.222] | CI includes 0 |
| bol | cond | 34 | +0.116 | [-0.048, +0.273] | CI includes 0 |
| rlt | plain | 37 | -0.011 | [-0.214, +0.228] | CI includes 0 |
| rlt | cond | 37 | -0.111 | [-0.322, +0.179] | CI includes 0 |
| bol+rlt | plain | 71 | +0.037 | [-0.107, +0.196] | CI includes 0 |
| bol+rlt | cond | 71 | +0.054 | [-0.094, +0.213] | CI includes 0 |

Stress-conditioned minus plain FPR, **high-stress band**, same fold and seed (15 pairs; Wilcoxon signed-rank):

| dataset | pairs | plain FPR mean | cond FPR mean | diff (mean ± std) | cond lower / higher / equal | Wilcoxon p |
|---|---|---|---|---|---|---|
| bol | 15 | 0.533 | 0.482 | -0.051 ± 0.181 | 5 / 4 / 6 | 0.3139 |
| rlt | 15 | 0.291 | 0.226 | -0.065 ± 0.169 | 3 / 1 / 11 | 0.1441 |
| bol+rlt | 15 | 0.477 | 0.425 | -0.051 ± 0.163 | 5 / 4 / 6 | 0.3139 |

Stress-conditioned minus plain FPR, **low-stress band**, same fold and seed (15 pairs; Wilcoxon signed-rank):

| dataset | pairs | plain FPR mean | cond FPR mean | diff (mean ± std) | cond lower / higher / equal | Wilcoxon p |
|---|---|---|---|---|---|---|
| bol | 15 | 0.507 | 0.385 | -0.122 ± 0.274 | 6 / 2 / 7 | 0.1235 |
| rlt | 15 | 0.296 | 0.324 | +0.029 ± 0.176 | 4 / 3 / 8 | 0.4974 |
| bol+rlt | 15 | 0.457 | 0.372 | -0.085 ± 0.234 | 6 / 3 / 6 | 0.2135 |

Predicted stress vs deception label (per clip, one fixed score per clip):

| dataset | clips | deceptive | truthful | mean stress (deceptive) | mean stress (truthful) | point-biserial r | r p-value | Mann-Whitney p | AUC of stress as deception score |
|---|---|---|---|---|---|---|---|---|---|
| bol | 325 | 162 | 163 | 0.213 | 0.111 | +0.196 | 0.0003701 | 0.2504 | 0.537 |
| rlt | 121 | 61 | 60 | 0.565 | 0.542 | +0.028 | 0.7628 | 0.9442 | 0.504 |
| bol+rlt | 446 | 223 | 223 | 0.309 | 0.227 | +0.115 | 0.01471 | 0.1076 | 0.544 |

Reading: no FPR difference between bands has a bootstrap CI excluding 0, and no paired conditioned-vs-plain difference has Wilcoxon p ≤ 0.05; the only nominally significant stress–label associations are the point-biserial correlations for BoL (r = +0.196, p = 0.00037) and for BoL+RLT combined (r = +0.115, p = 0.01471), but the Mann-Whitney tests on the same clips give p = 0.2504 and p = 0.1076 and the AUCs are 0.537 and 0.544 (RLT alone: r = +0.028, p = 0.7628, AUC 0.504). Caveats from `_summary.txt`: clips of one person and the three seeds are not independent (hence the group bootstrap), the stress score is a noisy MAS-trained model applied to other datasets, and the high-stress truthful bands are small (bol 81, rlt 30 clips).

## 6. Leakage ablation: person-grouped versus clip-level folds

Source: `deception/leakage_ablation.csv`. Late fusion, 5 folds × seeds 0,1,2 = **15 runs per cell**. `person` = the normal group-wise folds; `clip` = stratified 5-fold over clips (seed-42 shuffle), so the same speaker can be in train and test. Binomial p recomputed from `runs/within_*_predictions.csv` (person) and `runs/leakage_clip_predictions.csv` (clip).

| dataset | scheme | runs | seeds | accuracy | macro-F1 | ROC-AUC | binomial p — vs chance | share of test clips whose speaker is in train |
|---|---|---|---|---|---|---|---|---|
| rlt | person | 15 | 0,1,2 | 0.702 ± 0.067 | 0.677 ± 0.095 | 0.710 ± 0.081 | 9.815e-06 (n=121) — p<0.05 | 0.000 |
| rlt | clip | 15 | 0,1,2 | 0.755 ± 0.047 | 0.752 ± 0.048 | 0.835 ± 0.067 | 2.496e-08 (n=121) — p<0.05 | 0.592 |
| rlt | clip − person (difference of means) | — | — | +0.053 | +0.074 | +0.125 | — | — |
| bol | person | 15 | 0,1,2 | 0.548 ± 0.049 | 0.526 ± 0.052 | 0.575 ± 0.068 | 0.09594 (n=325) — NOT sig. above chance | 0.000 |
| bol | clip | 15 | 0,1,2 | 0.526 ± 0.045 | 0.509 ± 0.045 | 0.534 ± 0.051 | 0.3748 (n=325) — NOT sig. above chance | 1.000 |
| bol | clip − person (difference of means) | — | — | -0.021 | -0.017 | -0.041 | — | — |

The difference rows are differences of the two means (the CSV has no std for them). Clip-level folds raise all three metrics on RLT (accuracy +0.053, macro-F1 +0.074, ROC-AUC +0.125) but **lower** them on BoL (−0.021, −0.017, −0.041): the ablation does not show a consistent leakage gain. Neither BoL scheme is significantly above chance; on RLT both are. Every BoL test clip (975 of 975 over the 15 runs) has its speaker in the training folds under the clip scheme, RLT 215 of 363. Overlap is from `leakage_ablation_overlap.csv` summed over the 15 runs per scheme.

## 7. Error analysis (one line each)

- **Stress (Harshith):** *not available* — no stress error-analysis file exists in the repository or on any remote branch (listed under missing inputs).
- **Deception (`results/deception/errors_notes.md`, DRAFT):** the 5 most confident late-fusion errors per dataset (10 clips, mostly missed lies, 2 with mostly no face found) are listed, but the `Watched:` field is empty for all of them, so no findings yet.

## 8. Lighting robustness (BoL deception, brightness perturbation of the video)

Source: `deception/lighting.csv`. **One test fold (fold 0, 65 BoL clips) × seeds 0,1,2 = 3 runs**; video features are re-extracted at each brightness factor, audio features are the clean ones (so `concat` only sees perturbed video). `video` = video-only checkpoint, `concat` = the best fused checkpoint of that fold. Row `clean (stored features)` uses the original features; `x1.0 re-extracted` re-runs the extractor without a change. mean ± std over the 3 seeds (per-seed rows; they equal the CSV's `mean` rows). p = binomial vs chance from k = accuracy × 65 per seed, averaged over seeds.

| brightness | mode | runs | seeds | accuracy | macro-F1 | ROC-AUC | % frames with face | clips with no face | binomial p — vs chance |
|---|---|---|---|---|---|---|---|---|---|
| clean (stored features) | concat | 3 | 0,1,2 | 0.533 ± 0.024 | 0.526 ± 0.023 | 0.581 ± 0.052 | 99.98 | 0 | 0.6201 — NOT sig. above chance |
| clean (stored features) | video | 3 | 0,1,2 | 0.518 ± 0.032 | 0.509 ± 0.025 | 0.535 ± 0.027 | 99.98 | 0 | 0.8043 — NOT sig. above chance |
| x0.3 re-extracted | concat | 3 | 0,1,2 | 0.513 ± 0.054 | 0.498 ± 0.062 | 0.566 ± 0.055 | 69.79 | 4 | 1 — NOT sig. above chance |
| x0.3 re-extracted | video | 3 | 0,1,2 | 0.513 ± 0.049 | 0.505 ± 0.055 | 0.521 ± 0.034 | 69.79 | 4 | 1 — NOT sig. above chance |
| x0.5 re-extracted | concat | 3 | 0,1,2 | 0.497 ± 0.024 | 0.485 ± 0.009 | 0.549 ± 0.057 | 92.15 | 0 | 1 — NOT sig. above chance |
| x0.5 re-extracted | video | 3 | 0,1,2 | 0.451 ± 0.036 | 0.430 ± 0.009 | 0.485 ± 0.041 | 92.15 | 0 | 0.457 — NOT sig. above chance |
| x0.7 re-extracted | concat | 3 | 0,1,2 | 0.513 ± 0.009 | 0.503 ± 0.004 | 0.561 ± 0.071 | 99.50 | 0 | 1 — NOT sig. above chance |
| x0.7 re-extracted | video | 3 | 0,1,2 | 0.482 ± 0.032 | 0.463 ± 0.031 | 0.512 ± 0.030 | 99.50 | 0 | 0.8043 — NOT sig. above chance |
| x1.0 re-extracted | concat | 3 | 0,1,2 | 0.533 ± 0.024 | 0.526 ± 0.023 | 0.581 ± 0.052 | 99.98 | 0 | 0.6201 — NOT sig. above chance |
| x1.0 re-extracted | video | 3 | 0,1,2 | 0.518 ± 0.032 | 0.509 ± 0.025 | 0.535 ± 0.027 | 99.98 | 0 | 0.8043 — NOT sig. above chance |
| x1.3 re-extracted | concat | 3 | 0,1,2 | 0.549 ± 0.024 | 0.545 ± 0.023 | 0.588 ± 0.044 | 99.88 | 0 | 0.457 — NOT sig. above chance |
| x1.3 re-extracted | video | 3 | 0,1,2 | 0.523 ± 0.046 | 0.515 ± 0.041 | 0.549 ± 0.022 | 99.88 | 0 | 0.8043 — NOT sig. above chance |
| x1.6 re-extracted | concat | 3 | 0,1,2 | 0.533 ± 0.039 | 0.529 ± 0.039 | 0.577 ± 0.054 | 99.88 | 0 | 0.6201 — NOT sig. above chance |
| x1.6 re-extracted | video | 3 | 0,1,2 | 0.503 ± 0.032 | 0.494 ± 0.027 | 0.537 ± 0.039 | 99.88 | 0 | 1 — NOT sig. above chance |

Every cell is within chance range, including the unperturbed one, so this single-fold experiment cannot show a lighting effect on accuracy; the clearest effect is face detection at brightness 0.3 (69.79% of frames, 4 clips without any face) versus ≥ 99.50% at 0.7 and above.

## Source files used

All CSV/text files under `results/` were inspected; `*_predictions.csv` files were used only for the recomputed p-values and per-dataset fine-tuning metrics. `deception/runs/*_summary.csv`, `runs/*_runs.csv`, the `*_smoke*`-free per-run files and `run_all.log` were read for cross-checks only (they carry the same numbers as `deep.csv`). `leakage_ablation_overlap.csv` was used for the overlap column. The stress from-scratch audio row in section 1 comes from `stress/scratch_audio.csv` and `stress/runs/scratch_audio_runs.csv`.
