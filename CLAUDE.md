# Multimodal Stress and Deception Detection (course project)

## Hard rules
- NEVER run git commit, git push, git add, or any command that changes git history. The human reads `git diff` and commits by hand.
- NEVER put data, features, checkpoints, videos, or recordings in git. They live in data/, features/, features_perturbed/, checkpoints/ (all gitignored).
- Only edit the files the prompt names. Do not rewrite other people's files.
- Do not invent numbers. Every reported metric must come from a file in results/.

## Layout
data/<mas|bol|rlt>/, features/<dataset>/, src/common/ (shared), src/stress/, src/deception/, src/integration/, results/ (committed csv/png), docs/feature_contract.md

## Data contract (see docs/feature_contract.md)
- features/<dataset>/metadata.csv columns: clip_id, dataset, video_path, group_id, task, label, fold, duration_s, mas_labels
- folds: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on group_id. Never split by clip.
- features/<dataset>/<clip_id>.npz keys: audio_w2v (T,768) f16, audio_prosody (T,3) f32, audio_logmel (Tm,64) f16, video_face (T,768) f16, video_valid (T,) bool, video_flow (T,2) f32, video_crops64 (T,64,64) u8
- 5 steps per second (0.2 s) except audio_logmel (100 frames/s). Clips are cut to the first 60 s.
- Audio is 16 kHz mono. Scripts must skip clips already done and load an existing .npz before adding keys.

## Style
- Python 3, seeds fixed (0, 1, 2 for model runs), keep scripts short and readable.
- Print shapes and per-fold counts so the human can check.
