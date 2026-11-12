# Feature contract (shared by all four of us; change only by agreement)

## metadata: features/<dataset>/metadata.csv (one row per clip)
clip_id (unique in dataset, = .npz name) | dataset (mas, bol, rlt) | video_path | group_id (person for bol/rlt, source video for mas) | task (stress, deception) | label (0/1; stress 1 = stressed, deception 1 = deceptive) | fold (0-4, StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on group_id) | duration_s | mas_labels (MAS only, raw label string; empty otherwise)

## features: features/<dataset>/<clip_id>.npz
| key | shape, dtype | what | by |
| audio_w2v | (T_a,768) float16 | wav2vec2-base last hidden layer, mean over 0.2 s windows | P1 |
| audio_prosody | (T_a,3) float32 | pitch Hz (0 if unvoiced), RMS energy, voiced flag | P1 |
| audio_logmel | (T_m,64) float16 | 64-band log-mel, 100 frames/s | P1 |
| video_face | (T_v,768) float16 | face-expression ViT embedding | P3 |
| video_valid | (T_v,) bool | face found in frame | P3 |
| video_flow | (T_v,2) float32 | mean, std of optical-flow magnitude in face | P3 |
| video_crops64 | (T_v,64,64) uint8 | grayscale face crops | P3 |

All keys except audio_logmel are at 5 steps per second. Clips over 60 s are cut to the first 60 s.

## extra files for GPU fine-tuning
features/<dataset>/wav/<clip_id>.wav : 16 kHz mono, first 60 s
features/<dataset>/crops224/<clip_id>.npy : (16,224,224,3) uint8
