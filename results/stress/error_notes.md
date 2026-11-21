# Stress error analysis: 10 most confident mistakes (audio model, seed-averaged)

Notes written after watching each clip. "Label check" is my own judgement of the MAS label.

## False positives (label: no stress, model: stressed)
1. Skinhead_interview-Donny: calm confession of an incident. Label check: agree.
2. Addicted_To_The_Streets-Easy: speaker under the influence. Label check: arguable.
3. Keep it 100 (Ghost): calm, relaxed explanation of the place they are in. Label check: agree.
4. Keep it 100 (zodiac sign): high-pitched, confident opinion of a woman. Label check: agree.
5. Keep it 100 (Ghost): talk about an experience, deep voice. Label check: agree.

## False negatives (label: stress, model: no stress)
6. Perseverance_and_Resilience-Pam: calm, collected, relaxed account of emotional abuse by her mother; seems to have got through it. Label (audio stress only): arguable; the stress is in the content, not the delivery.
7. Crystal_Meth_Addict_interview-Stacy: weeping while narrating a sad incident. Label check: agree.
8. Crystal_Meth_Addict_interview-Stacy: describing an incident, tearing up lightly (same woman as 7). Label check: agree.
9. Heroin_Addict-Caroline: tearing up while telling the camera "you can rely on me". Label (face stress, prosodic stress): arguable; may be emotion rather than stress.
10. Whats_Your_Biggest_Anxiety (Keep it 100): fast talk, seems tense, talking about anxiety disorder. Label check: agree. Very short clip (2.9 s).

## Observations (from 10 clips, so hypotheses, not findings)
- Three of five false negatives are people crying or tearing up (7, 8, 9): visible distress that the audio model scores below 1%.
- Clip 6's stress is in what she says, not how she sounds, which an audio-only model cannot see.
- The five false positives share no single acoustic cue (calm, high-pitched, deep, intoxicated) but all score about 0.95. Three come from the "Keep it 100" series, so a show-level effect is possible.
- Folds group by source video, not by show.
