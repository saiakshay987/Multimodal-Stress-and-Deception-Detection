# Multimodal Stress and Deception Detection

IDSE course project. We fuse **audio** (wav2vec2, prosody) and **video** (face-expression ViT, optical flow) to
detect **stress** on MAS and **deception** on Bag-of-Lies (BoL) and Real-Life Trial (RLT).
The deception output is a **cue for human review, never a verdict**: it must not be used to decide that a person lied.

## Folder layout

```
src/common/        shared code: metadata, feature extraction, check_features, data, models, train, finetune
src/stress/        stress on MAS: baseline, run_all, scratch_audio, train_final, predict_stress
src/deception/     deception on BoL/RLT: baseline, run_all, report, scratch_video, finetune_pooled,
                   compare_finetune, robustness
src/integration/   train_multitask, stress_aware, transfer (stress + deception together)
results/           CSVs and plots written by the scripts (in git)
docs/              feature_contract.md (metadata columns and .npz keys)
data/  features/  checkpoints/   NOT in git (raw videos, extracted features, model weights)
```

## Setup

- Python 3.10, PyTorch. CUDA is used when available, otherwise CPU. Target GPU: 6 GB RTX 4050 (fp16 autocast).
- `pip install -r requirements.txt` (`peft` is required for LoRA fine-tuning; `ffmpeg` is needed by `extract_audio.py`).
- **The datasets are not included.** Put them under `data/` yourself. TODO: where to obtain MAS, BoL and RLT, and the
  expected folder layout of `data/bol` and `data/rlt`.
- Run every command from the repository root. Folds always come from the `fold` column of
  `features/<dataset>/metadata.csv` (split by `group_id`, never by clip). Seed 42 unless a seed is passed.

## How to reproduce

```bash
# 1. Metadata (features/<dataset>/metadata.csv). RLT needs a hand-filled features/rlt/speaker_ids.csv
#    (or --speaker_ids PATH).
python src/common/make_metadata.py mas data/mas
python src/common/make_metadata.py bol data/bol
python src/common/make_metadata.py rlt data/rlt

# 2. Feature extraction (per dataset: mas bol rlt). extract_video needs the MediaPipe face model at
#    checkpoints/mediapipe/blaze_face_short_range.tflite (default --face_model). TODO: how to obtain that file.
python src/common/extract_audio.py mas
python src/common/extract_video.py mas

# 3. Check the features against docs/feature_contract.md
python src/common/check_features.py mas bol rlt

# 4. Stress fusion on MAS: pooled-feature baseline, then audio/video/concat/late/xattn, 5 folds x seeds 0 1 2
python src/stress/baseline.py
python src/stress/run_all.py
python src/stress/scratch_audio.py              # from-scratch audio CNN on log-mel

# 5. Deception runs: pooled-feature baseline, 5 settings x 5 modes x 3 seeds, then the summary table and plot
python src/deception/baseline.py
python src/deception/run_all.py                 # add --dry-run to only print the plan
python src/deception/report.py
python src/deception/scratch_video.py           # from-scratch video CNN+GRU on 64x64 crops

# 6. LoRA fine-tuning of wav2vec2 + ViT (needs wav/ and crops224/; encoders must be in the HF cache,
#    or pass --allow-download)
python src/common/finetune.py --task stress --datasets mas
python src/deception/finetune_pooled.py --task deception --datasets bol rlt --balance dataset --modes xattn audio video
python src/deception/compare_finetune.py
python src/common/finetune.py --task stress --datasets mas --smoke   # optional quick check, writes *_smoke files

# 7. Multitask, stress-aware deception, transfer (each also has --smoke and --compare-only)
python src/integration/train_multitask.py
python src/integration/stress_aware.py
python src/integration/transfer.py

# 8. Final stress model and scoring (audio-only model trained on all MAS clips; CPU is fine for prediction)
python src/stress/train_final.py --epochs 7 --out checkpoints/stress_final.pt
python src/stress/predict_stress.py --datasets bol rlt --out results/stress/stress_scores_on_deception.csv
python src/stress/predict_stress.py --clips features/bol/User_0_run_0.npz

# 9. Final deception model and predict_deception.py
#    TODO: no final-training script or predict_deception.py exists in src/ yet.

# Optional: lighting robustness of the BoL models (one fold)
python src/deception/robustness.py --ckpt_dir checkpoints/deception/within_bol   # TODO: confirm checkpoint folder
```

## Where the results live (all under `results/`)

- Stress baseline: `stress/baseline.csv`. Fusion runs: `stress/deep.csv`, `stress/deep_runs.csv`,
  `stress/deep_predictions.csv`, `stress/fusion_comparison.png`. Scratch audio: `stress/runs/scratch_audio_runs.csv`.
- Stress fine-tuning: `stress/finetune.csv`, `stress/finetune_summary.csv`, `stress/finetune_predictions.csv`.
- Stress scores on deception clips: `stress/stress_scores_on_deception.csv`.
- Deception baseline: `deception/baseline.csv`. Fusion summary: `deception/deep.csv`, `deception/fusion_comparison.png`.
  Per-setting runs: `deception/runs/{within_bol,within_rlt,pooled,bol_to_rlt,rlt_to_bol}.csv` (+ `_summary`, `_predictions`).
- Deception scratch video: `deception/scratch_video.csv`. Fine-tuning: `deception/finetune.csv`,
  `deception/finetune_summary.csv`. Lighting: `deception/lighting.csv`, `deception/lighting.png`.
- Integration: `integration/multitask.csv` and `multitask_vs_single.csv`; `integration/stress_aware.csv` and
  `stress_aware_vs_plain.csv`; `integration/transfer.csv` and `transfer_vs_scratch.csv`.
- Files named `*_smoke*` are quick-check outputs, not results.

## Team and contributions

<!-- TODO: team members, roles and who did what (feature extraction, stress, deception, integration). -->
<!-- TODO: course name/term, supervisor, dataset citations and licences, how to cite this work. -->

## Limitations

- The datasets are small (MAS: grouped by source video; BoL 325 and RLT 121 test clips in `results/deception/deep.csv`),
  and results vary a lot between folds and seeds.
- Accuracy is near chance for most deception settings, and cross-dataset transfer between BoL and RLT does not work
  (see `results/deception/deep.csv`, `bol_to_rlt` and `rlt_to_bol` rows).
- On MAS, video alone is at chance for stress (`results/stress/deep.csv`: video accuracy_mean 0.494 over 15 runs).
- Stress scores shift between datasets: the final MAS model gives clearly different score distributions on BoL
  and RLT clips (`results/stress/stress_scores_on_deception.csv`), so its probabilities are not comparable across datasets.
- No real-world validation. Deception scores are a cue for a human reviewer, never evidence that someone lied.
