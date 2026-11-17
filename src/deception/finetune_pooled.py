#!/usr/bin/env python3
"""Run src/common/finetune.py with dataset-balanced training batches (pooled Bag-of-Lies + Real-Life Trial).

src/common/finetune.py has no balanced sampler (its train loader is a plain shuffle) and src/common/ may
not be edited, so this wrapper imports finetune.py unchanged and replaces, at run time, only
  * ``finetune.build_raw_loader``: for the training loader it draws batches with
    ``data.BalancedBatchSampler`` (the same sampler train.py uses for ``--balance dataset``); validation
    and test loaders are untouched;
  * ``train.setup_id``: appends ";balance=<mode>" so the rows can be told apart from unbalanced ones
    (the setup string ends in ";balance=dataset;ft=lora8" once finetune.py adds its own suffix).
Everything else (model, optimiser, splits by fold column, early stopping, outputs, skipping of
finished runs) is finetune.py itself, and all of its command-line arguments are accepted.

Example (from the repository root):
    python src/deception/finetune_pooled.py --task deception --datasets bol rlt --balance dataset \
        --modes xattn audio video --folds 0 1 2 3 4 --seeds 0 --out results/deception/finetune.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import finetune as F  # noqa: E402  (src/common/finetune.py, unchanged)
import train as T  # noqa: E402
from data import BalancedBatchSampler  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def install(balance_by, label):
    """Patch finetune's train loader and the setup string; validation / test loaders stay as they are."""
    plain_loader, plain_setup_id = F.build_raw_loader, T.setup_id

    def build_raw_loader(rows, root, batch_size, train, seed, workers=0, device=None):
        if not train:
            return plain_loader(rows, root, batch_size, train, seed, workers, device)
        ds = F.RawClipDataset(rows, root, train=True, seed=seed)
        return DataLoader(ds, batch_sampler=BalancedBatchSampler(ds.rows, batch_size, balance_by, seed=seed),
                          collate_fn=F.collate_raw, num_workers=workers, persistent_workers=workers > 0,
                          pin_memory=device is not None and device.type == "cuda")

    F.build_raw_loader = build_raw_loader
    T.setup_id = lambda *a, **k: plain_setup_id(*a, **k) + f";balance={label}"


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--balance", choices=list(T.BALANCE_CHOICES), default="dataset")
    known, rest = ap.parse_known_args(argv)
    if "-h" in rest or "--help" in rest:
        print(__doc__)
        print("--balance {%s}  (default dataset)\n" % ",".join(T.BALANCE_CHOICES))
        F.parse_args(["--help"])
    if T.BALANCE_CHOICES[known.balance]:
        install(T.BALANCE_CHOICES[known.balance], known.balance)
    F.main(rest)


if __name__ == "__main__":
    main()
