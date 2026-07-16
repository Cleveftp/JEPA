# JEPA "Second" model -- build & training summary

Second training run of the from-scratch I-JEPA (CuPy). Same architecture as
First; this run explores **harder masking, a slower EMA teacher, and 3x longer
training**. It is the best model so far on the downstream metrics.

## What changed vs First

| knob | First | Second | why |
|---|---|---|---|
| mask keep-ratio | 0.8 (12 visible / 4 masked) | **0.6 (~9 visible / ~7 masked)** | harder prediction task -> richer reps |
| masking helper | random (`mask_tiles`) | `block_mask_tiles` added | contiguous block masking, closer to real I-JEPA |
| EMA_ratio (per step) | 1e-3 | **5e-5** | slower, more stable target (per-step rate compounds over steps) |
| epochs | 100 | **300** | |
| batch_size | none (full per-sample) | 100 | |
| lr / token_dim / heads | 0.003 / 64 / 4 | same | |

Architecture (teacher/student/predictor, transformer block, final LayerNorm,
absolute positional encodings, summed mask-token gradient) is identical to
First -- see `Models/First/summary.md` for the full component description.

## Result snapshot (final, e300)

| readout | value | vs First e100 |
|---|---|---|
| linear probe, concat 16 patches (1024-d) | **0.973** | 0.953  (+0.020) |
| linear probe, mean-pool (64-d) | 0.922 | 0.913  (+0.009) |
| effective dim | **7.5 / 64** | 6.1    (+1.4) |
| patch diversity | 0.287 | 0.351  (-0.064) |
| silhouette (KMeans-10) | 0.237 | 0.237  (tie) |

Concat linear is now at/above the ~0.97 raw-pixel ceiling, and effective dim is
the highest of any run -- the reps use more of their capacity. The one regression
is patch diversity (tokens slightly more redundant), though it recovered from a
mid-training trough (see below).

## Progression (mean-pool lin / concat lin / patch div / eff dim / silhouette)

| epoch | mean | concat | pdiv | effdim | sil |
|------:|:----:|:------:|:----:|:------:|:---:|
| 0   | 0.833 | 0.958 | 0.345 | 5.9 | 0.224 |
| 50  | 0.851 | 0.967 | 0.180 | 3.2 | 0.295 |
| 100 | 0.880 | 0.962 | 0.224 | 3.7 | 0.258 |
| 150 | 0.896 | 0.971 | 0.260 | 5.0 | 0.284 |
| 200 | 0.913 | 0.967 | 0.272 | 5.9 | 0.270 |
| 250 | 0.927 | 0.967 | 0.277 | 6.7 | 0.256 |
| 300 | 0.922 | 0.973 | 0.287 | 7.5 | 0.237 |

Read: same "compress then organize" shape as First, but deeper and stretched
over 3x the epochs. Patch diversity bottoms at e50 (0.18) -- a trough, not
collapse -- then climbs back to 0.287. The early silhouette spike (0.295 at e50)
is an artifact of that compressed phase and normalizes as diversity recovers.
Sweet spot is ~e250-e300 (e250 best mean-pool 0.927; e300 best concat + eff dim).

## Files in this folder (mirrors Models/First)

- `Checkpoints/teacher_e{0..300}.npz` -- teacher weights every 10 epochs.
- `JEPA.py`, `extract_features.py`, `save_util.py` -- code snapshots for this run.
- `output.json` -- per-epoch training/val MSE + teacher-magnitude log.
- `train_reps.npy` (1707,16,64), `test_reps.npy` (90,16,64) -- exported reps.
- `teacher.npz` -- final save (== e300).

## Reproduce / load

```python
ffn1 = Layer(4, 32); ffn2 = Layer(32, 64, 'linear')
p1 = Parameter((16, 64)); trans1 = Transformer(64)      # n_heads=4
load_model((ffn1, ffn2, p1, trans1), "Models/Second/Checkpoints/teacher_e300.npz")
# forward: ffn1 -> ffn2 -> +p1 -> trans1 -> LayerNorm  (see extract_features.py)
```

Verdict: best reps to date (concat 0.973, eff dim 7.5), at the cost of a little
patch diversity and 3x the compute -- diminishing returns on a dataset already
saturated near the pixel ceiling, but a clean win on representation richness.
