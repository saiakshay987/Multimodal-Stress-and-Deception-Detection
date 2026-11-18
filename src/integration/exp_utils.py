"""Small helpers shared by stress_aware.py and transfer.py (everything else comes from
src/common/train.py and src/integration/train_multitask.py, which are imported, not copied)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from train import METRICS, compute_metrics  # noqa: E402


def assert_disjoint(train_df, val_df, test_df):
    """Train / val / test must never share a group (person or source video)."""
    parts = {"train": train_df, "val": val_df, "test": test_df}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (set(parts[a]["group_key"]) & set(parts[b]["group_key"])), f"{a}/{b} share a group"


def eval_rows(preds, eval_sets, task_of_dataset, setup, mode, fold, seed):
    """One result row per test set in ``eval_sets`` = [(name, [datasets]), ...]."""
    rows = []
    for name, members in eval_sets:
        d = preds[preds["dataset"].isin(members)]
        if d.empty:
            continue
        rows.append({"setup": setup, "mode": mode, "fold": fold, "seed": seed, "test_dataset": name,
                     "task": task_of_dataset[members[0]], "n_test": len(d), **compute_metrics(d["label"], d["prob"])})
    return rows


def summarize(results, summary_csv):
    """Mean / std over runs per test_dataset; written to ``summary_csv`` and printed."""
    summary = (results.groupby("test_dataset", sort=False)
               .agg(n_runs=("fold", "size"), **{f"{m}_{s}": (m, s) for m in METRICS for s in ("mean", "std")},
                    time_s_mean=("time_s", "mean")).reset_index())
    summary.to_csv(summary_csv, index=False)
    for r in summary.itertuples():
        print(f"  {r.test_dataset:8s} (n={r.n_runs})  " + "  ".join(
            f"{m} {getattr(r, m + '_mean'):.3f} ± {getattr(r, m + '_std'):.3f}" for m in METRICS))
    return summary


def clip_sets(pred_df):
    """{(dataset, fold, seed): frozenset(clip_id)} of a predictions table."""
    return {k: frozenset(g["clip_id"]) for k, g in pred_df.groupby(["dataset", "fold", "seed"])}


def paired_table(new, base, test_sets, new_name, base_name, new_pred=None, base_pred=None, smoke=False):
    """Paired difference (new - base) per test set and metric, matched on (fold, seed, test_dataset).

    ``new`` / ``base``: per-run tables with fold, seed, test_dataset, n_test and the METRICS.
    ``new_pred`` / ``base_pred``: optional per-clip prediction tables (clip_id, dataset, fold, seed) used to check
    that both sides were tested on exactly the same clips. Prints the table and returns it as a DataFrame.
    """
    ok = True
    for ds in test_sets:
        a, b = new[new["test_dataset"] == ds], base[base["test_dataset"] == ds]
        ka, kb = set(zip(a["fold"], a["seed"])), set(zip(b["fold"], b["seed"]))
        if ka != kb and not smoke:
            ok = False
            print(f"  !! {ds}: (fold, seed) only in {new_name}: {sorted(ka - kb)[:6]}; only in {base_name}: {sorted(kb - ka)[:6]}")
    m = new.merge(base, on=["fold", "seed", "test_dataset"], suffixes=("_new", "_base"))
    m = m[m["test_dataset"].isin(test_sets)]
    bad = m[m["n_test_new"] != m["n_test_base"]]
    if len(bad) and not smoke:
        ok = False
        print(f"  !! {len(bad)} paired runs have different n_test, e.g. "
              f"{bad[['test_dataset', 'fold', 'seed', 'n_test_new', 'n_test_base']].head(3).values.tolist()}")
    if new_pred is not None and base_pred is not None:
        c_new, c_base = clip_sets(new_pred), clip_sets(base_pred)
        diff = [k for k in c_new if k in c_base and c_new[k] != c_base[k]]
        if diff and not smoke:
            ok = False
            print(f"  !! test clip sets differ in {len(diff)} (dataset, fold, seed) cells, e.g. {diff[:3]}")
        if smoke:
            print(f"  (smoke) test clip sets differ in {len(diff)} of {len(c_new)} cells, as expected for subsampled clips")
        elif ok:
            print(f"  OK: same folds, seeds, n_test and test clip ids in all {len(c_new)} (dataset, fold, seed) cells")
    elif ok and not smoke:
        print("  OK: same folds, seeds and n_test in every pair (clip ids not checked)")

    rows = []
    for ds in test_sets:
        g = m[m["test_dataset"] == ds]
        for metric in METRICS:
            d = (g[metric + "_new"] - g[metric + "_base"]).dropna()
            rows.append({"test_dataset": ds, "metric": metric, "n_pairs": len(d),
                         f"{base_name}_mean": g.loc[d.index, metric + "_base"].mean(),
                         f"{new_name}_mean": g.loc[d.index, metric + "_new"].mean(),
                         "diff_mean": d.mean() if len(d) else np.nan,
                         "diff_std": d.std(ddof=1) if len(d) > 1 else np.nan,
                         "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()),
                         "n_equal": int((d == 0).sum())})
    table = pd.DataFrame(rows)
    print(f"  {'test set':8s} {'metric':9s} {'n':>3s} {base_name[:12]:>12s} {new_name[:12]:>12s}   "
          f"diff mean ± std   (better/worse/equal runs)")
    for r in table.itertuples():
        print(f"  {r.test_dataset:8s} {r.metric:9s} {r.n_pairs:3d} {getattr(r, base_name + '_mean'):12.3f} "
              f"{getattr(r, new_name + '_mean'):12.3f}   {r.diff_mean:+.3f} ± {r.diff_std:.3f}      "
              f"({r.n_better}/{r.n_worse}/{r.n_equal})")
    print(f"  diff = {new_name} - {base_name} over paired (fold, seed) runs; AUC pairs are fewer where a test fold "
          "has one class.")
    return table
