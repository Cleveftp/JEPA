# JEPA "First" model -- build & training summary

A from-scratch **I-JEPA-style** self-supervised model, implemented by hand in
**CuPy** (custom NN library, hand-derived + gradient-checked backprop). Trained
on the sklearn `load_digits` set (8x8 grayscale digits). This document records
exactly how the model in this folder was built and trained, so it can be
reverted to or reproduced.

## Result snapshot (final, e100)

| metric | value | notes |
|---|---|---|
| linear probe acc | 0.913 | mean-pooled 64-d rep, 25% held-out |
| kNN(10) acc | 0.864 | same |
| linear probe acc (concat 16 patches, 1024-d) | 0.953 | no retrain -- just don't mean-pool |
| effective dim | 6.1 / 64 | recovered from an e10 trough of 2.7 |
| silhouette (KMeans-10) | 0.237 | up 74% from random init |
| raw-pixel linear baseline | ~0.97 | ceiling for this toy dataset |

Full per-epoch numbers are in `progression.csv`; the collage is
`checkpoint_progression.png`.

## Data pipeline

- `load_digits` -> `X` reshaped to `(N, 8, 8)`, pixel values divided by 16 (range ~0..1).
- Train/test split `test_size=0.05` (no fixed seed).
- **Tiling:** each 8x8 image is split into `2x2` patches -> **16 patches** of 4 pixels each, flattened to `(16, 4)`. (`Embedding._tile`; the `Embedding` object is used only for tiling here -- the content-hash token store was not used in the final training loop.)

## Architecture

Token width `token_dim = 64`, sequence length `stack_dim = 16`.

### Shared building blocks (NeuralNetwork/ + Transformer/)
- **Layer**: dense `y = x @ W.T + b`, activation relu/linear; single- and 2-D (sequence) inputs; SGD update inside `backward`.
- **LayerNormalization**: normalizes over the last axis (no learnable gamma/beta), eps 1e-5.
- **Parameter**: additive table `x + P`, optionally indexed `x + P[idx]` (used for positional encodings).
- **Simple_Parameter**: a bare learnable vector (used for the predictor mask token).
- **Transformer block** (pre-norm): `x = x + MHA(LN(x))`, then `x = x + FFN(LN(x))`, where FFN = `Layer(64,256,relu) -> Layer(256,64,linear)`.
- **multi_head_attention**: 4 heads, head_dim 16; per-head `self_attn` (wQ/wK/wV: 64x16), outputs concatenated and mixed by `wO` (64x64).
- **self_attn**: scaled dot-product `softmax(QK^T / sqrt(16)) V`, all weight grads + input grad hand-derived (verified by finite-difference gradient checks in tests/claude).

### Teacher  (ffn1, ffn2, p1, trans1, teach_ln) -- EMA only, no backprop
`tiles(16,4) -> Layer(4,32,relu) -> Layer(32,64,linear) -> + p1[all 16 positions] -> Transformer -> final LayerNorm`.
Encodes ALL 16 patches. Its output is the prediction target (stop-gradient).

### Student  (ffn3, ffn4, p2, trans2, stud_ln) -- trained
Same shape as the teacher, but encodes ONLY the visible (unmasked) patches.
Positional table `p2` is indexed by absolute patch position (`visible_idx`), so
"position k" means the same patch regardless of which are masked.

### Predictor  (masked_p, p3, trans3, pred_ln) -- trained
- `masked_p`: one shared learnable **mask token** (64-d).
- `collate_masked_with_unmasked`: builds the full 16-slot sequence -- student reps at visible positions, the mask token broadcast into masked positions.
- `+ p3` absolute positional encoding (its own table), then Transformer, then final LayerNorm.
- The predictor's outputs at the masked positions are the predictions.

## Masking

- `ratio = 0.8` is the fraction **kept visible**: 12 visible patches, **4 masked** per image.
- `mask_tiles` draws a fresh random `visible_idx` / `masked_idx` **per image, every step**, and returns the visible tiles.

## Objective & backward (three parts)

Loss = MSE between predictor output and teacher output, **at the masked positions only**:
`d_full` is zeros except masked rows = `2 * (p_out - t_out) / token_dim`.

1. **Predictor backward**: `d_full -> pred_ln -> trans3 -> p3`; the **mask token** receives the **sum** of gradients over all masked rows.
2. **Student backward**: the gradient that lands on the **visible** rows (`d_seq[visible_idx]`) flows into the student encoder. (The student is trained indirectly -- gradient reaches it through the predictor's attention, never by encoding masked patches.)
3. **Teacher**: never receives gradient.

## EMA teacher

- Teacher initialized as a **hard copy** of the student (`update_teacher(..., beta=1.0)` once before training).
- Then per-step EMA `teacher = (1-b)*teacher + b*student` with **`EMA_ratio = 0.001`**.
- `update_teacher` recurses through every weight array: both dense layers (W,B), the positional table (P), and the full transformer (mha `wO` + each head's wQ/wK/wV + the two FFN layers). LayerNorms have no params.

## Hyperparameters (this run)

| name | value |
|---|---|
| optimizer | vanilla SGD (`w -= lr * dW`, no momentum) |
| lr | 0.003 |
| epochs | 100 |
| batching | none (single image per step) |
| token_dim | 64 |
| stack_dim (patches) | 16 |
| mask keep-ratio | 0.8 (12 visible / 4 masked) |
| EMA_ratio | 0.001 per step |
| final rep | mean over 16 patch tokens |

## Key decisions / bug fixes that made it work

1. **Final LayerNorm on target and prediction.** Without it the representation
   scale ran away (teacher magnitude climbing, MSE feeding back on itself),
   diverging around epoch ~50. Normalizing the target removed the scale degree
   of freedom and stabilized training. This is the single most important fix.
2. **EMA rate tuned to steps-per-epoch.** A per-step EMA compounds over the
   number of steps; at full-dataset size the teacher was tracking the student
   within one epoch and destabilizing. Lowered to 0.001 for a slow, lagging
   target.
3. **Absolute positional encodings.** Positions are keyed by patch slot (0..15),
   consistent across teacher/student; the predictor has its own table. This is
   what lets the mask tokens know *which* patch they are predicting.
4. **Mask-token gradient = sum over masked slots** (it is one shared vector
   broadcast to every masked position).

## Progression (per epoch)

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

## Files in this folder

- `Checkpoints/teacher_e{0..100}.npz` -- teacher weights every 10 epochs (`save_util.save_model` format: dense W/B, positional P, mha wO + per-head wQ/wK/wV, transformer ffn W/B).
- `extract_features.py` -- loads a checkpoint and exports per-patch reps (`train_reps.npy`, `test_reps.npy`); applies the final LayerNorm.
- `save_util.py` -- `save_model` / `load_model` for the teacher module tuple.
- `train_reps.npy`, `test_reps.npy` -- exported representations.
- `output.json` -- per-epoch training/val MSE + teacher-magnitude log.
- `progression.csv`, `checkpoint_progression.png`, `rep_quality_e10.png` -- downstream eval artifacts.

## How to revert / reproduce

Load any checkpoint with the same module shapes and `save_util.load_model`:

```python
ffn1 = Layer(4, 32); ffn2 = Layer(32, 64, 'linear')
p1 = Parameter((16, 64)); trans1 = Transformer(64)   # n_heads=4
load_model((ffn1, ffn2, p1, trans1), "Models/First/Checkpoints/teacher_e100.npz")
# forward: ffn1 -> ffn2 -> +p1 -> trans1 -> LayerNorm  (see extract_features.py)
```

The teacher at `teacher_e100.npz` is the final "First" model.
