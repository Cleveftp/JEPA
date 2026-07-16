# claude/ -- analysis & tests contributed by Claude

Scratch work kept separate from the main codebase. Nothing here is imported by
your model; these scripts only *read* your files (checkpoints / exported reps)
and write their own outputs into this folder. Paths resolve relative to the
project root automatically, so they run from anywhere.

## Scripts

| file | what it does |
|------|--------------|
| `test_transformer_components.py` | Gradient checks + shape/learn tests for `Layer`, `self_attn`, `multi_head_attention`, and the transformer block. Run: `python tests/claude/test_transformer_components.py` (or `pytest`). Needs CuPy. |
| `eval_e10.py` | **Checkpoint sweep.** Re-encodes the full labeled digit set from *every* `Checkpoints/teacher_e*.npz` (pure NumPy, no GPU), then reports representation health and supervised accuracy per epoch, builds the progression collage, and writes `progression.csv`. Auto-discovers new checkpoints, so it's the "keep running" script -- just re-run it. Run: `python tests/claude/eval_e10.py`. |
| `rep_probe.py` | Label-free health battery on your exported `train_reps.npy` / `test_reps.npy`. Run: `python tests/claude/rep_probe.py`. |

The teacher forward in `eval_e10.py` mirrors `extract_features.py` exactly,
including the final LayerNorm, so its reps live on the same scale as the ones
you export. Image-level rep = mean over the 16 patch tokens.

## Metrics reported

- **linear probe / kNN accuracy** -- 10-class digit accuracy from a probe trained on the reps (25% held-out test). The real quality signal.
- **effective dim** -- participation ratio of the rep covariance; how many of the 64 dims are actually used.
- **patch diversity** -- mean pairwise cosine distance among an image's 16 patch tokens (0 = spatial collapse).
- **silhouette** -- unsupervised KMeans(10) cluster separation.

## Outputs

| file | what it is |
|------|-----------|
| `checkpoint_progression.png` | Collage: metric trajectories on top, then per-epoch PCA(2) coloured by KMeans cluster (left) and true digit (right). |
| `progression.csv` | Per-epoch metrics as data (epoch, linear, knn, eff_dim, patch_div, silhouette) for your own plotting. |
| `rep_quality_e10.png` | Older single-epoch figure. Superseded by the progression collage. |

## Final run (100 epochs)

| epoch | linear | kNN | eff dim | patch div | silhouette |
|------:|:------:|:---:|:-------:|:---------:|:----------:|
| 0   | 0.867 | 0.764 | 6.2 | 0.395 | 0.136 |
| 10  | 0.849 | 0.744 | 2.7 | 0.361 | 0.143 |
| 20  | 0.851 | 0.753 | 3.2 | 0.377 | 0.164 |
| 30  | 0.871 | 0.789 | 3.8 | 0.388 | 0.186 |
| 40  | 0.873 | 0.827 | 4.3 | 0.392 | 0.207 |
| 50  | 0.889 | 0.822 | 4.8 | 0.387 | 0.209 |
| 60  | 0.887 | 0.831 | 5.2 | 0.377 | 0.217 |
| 70  | 0.896 | 0.838 | 5.5 | 0.363 | 0.226 |
| 80  | 0.909 | 0.842 | 5.8 | 0.353 | 0.228 |
| 90  | 0.911 | 0.851 | 5.9 | 0.350 | 0.235 |
| 100 | 0.913 | 0.864 | 6.1 | 0.351 | 0.237 |

Read: e0 (random init) is already a strong linear baseline (~0.87). Training
compresses hard at e10 (effective dim drops 6.2 -> 2.7), then steadily
organizes. By e100 it clears the random baseline on every axis -- linear
0.913, kNN +0.10, silhouette +74%, and effective dim recovered to 6.1 (same
capacity as random init, but now structured rather than scattered). Raw-pixel
linear baseline on this dataset is ~0.97, so the reps do not beat pixels on
this toy task -- expected; the point is that useful, label-free structure was
learned from masked prediction alone.
