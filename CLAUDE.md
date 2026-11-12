# Project rules for Claude Code

## Git
- Never run git commit, git push, git merge, git rebase, git reset, or git stash.
- Never create, switch or delete branches.
- Leave every change uncommitted. A teammate reviews the diff and commits it.
- At the end of each task, list the files you created or changed.

## Data
- Never move, modify or delete anything under data/; only read it.
- Never download datasets or upload anything anywhere.
- Write derived files only under features/, features_perturbed/, results/ and checkpoints/.

## Code
- Python 3.10 and PyTorch. Use CUDA when available and fall back to CPU.
- Our GPUs are RTX 4050 laptop GPUs with 6 GB of memory: use fp16 autocast, small batches with gradient accumulation, and gradient checkpointing for pretrained encoders. Print peak GPU memory in every new training script.
- Every script takes paths as command-line arguments; no hard-coded absolute paths.
- Fix random seeds (42 unless a seed is passed) everywhere.
- Splits are always by group_id (person or source video), never by clip. Always use the fold column in features/<dataset>/metadata.csv.
- Do not change files in src/common/ without being asked: both teams depend on them.

## Data contract
- The metadata columns and .npz feature keys are defined in docs/feature_contract.md. Follow it exactly.
