# Most confident errors: within-dataset late fusion (DRAFT)

Source: results/deception/runs/within_{bol,rlt}_predictions.csv, mode=late, prob averaged over seeds 0-2.
Wrong = (mean prob >= 0.5) != label. Ranked by |prob - label|. video_path is relative to data/.
"video_valid mostly False" = fraction of frames with a face found < 0.5 (fraction in brackets).
Label 1 = deceptive, so label 1 / low prob is a false negative and label 0 / high prob is a false positive.

## Bag-of-Lies

1. User_17_run_8 | bol | label=1 | mean_prob=0.006 | bol/Finalised/User_17/run_8/video.mp4 | video_valid mostly False: no (1.00) | Watched:
2. User_21_run_3 | bol | label=0 | mean_prob=0.994 | bol/Finalised/User_21/run_3/video.mp4 | video_valid mostly False: no (1.00) | Watched:
3. User_17_run_6 | bol | label=1 | mean_prob=0.010 | bol/Finalised/User_17/run_6/video.mp4 | video_valid mostly False: no (1.00) | Watched:
4. User_5_run_2 | bol | label=0 | mean_prob=0.990 | bol/Finalised/User_5/run_2/video.mp4 | video_valid mostly False: YES (0.38) | Watched:
5. User_2_run_1 | bol | label=1 | mean_prob=0.011 | bol/Finalised/User_2/run_1/video.mp4 | video_valid mostly False: no (1.00) | Watched:

## Real-Life Trial

1. trial_lie_044 | rlt | label=1 | mean_prob=0.011 | rlt/Clips/Deceptive/trial_lie_044.mp4 | video_valid mostly False: no (0.98) | Watched:
2. trial_lie_052 | rlt | label=1 | mean_prob=0.022 | rlt/Clips/Deceptive/trial_lie_052.mp4 | video_valid mostly False: no (0.85) | Watched:
3. trial_lie_045 | rlt | label=1 | mean_prob=0.050 | rlt/Clips/Deceptive/trial_lie_045.mp4 | video_valid mostly False: YES (0.00) | Watched:
4. trial_lie_035 | rlt | label=1 | mean_prob=0.052 | rlt/Clips/Deceptive/trial_lie_035.mp4 | video_valid mostly False: no (1.00) | Watched:
5. trial_truth_006 | rlt | label=0 | mean_prob=0.938 | rlt/Clips/Truthful/trial_truth_006.mp4 | video_valid mostly False: no (0.77) | Watched:
