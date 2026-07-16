import cupy as cp

def mask_tiles(tiles, ratio=0.8):
    n = len(tiles)
    visible_idx = cp.random.choice(n, size=int(n*ratio), replace=False)
    masked_idx  = cp.setdiff1d(cp.arange(n), visible_idx)   # Cool function to select everything not chosen in a 1D array!
    return tiles[visible_idx], visible_idx, masked_idx

def block_mask_tiles(tiles, grid=(4, 4), block=(2, 2)):
    # mask blocks instead of random
    gh, gw = grid
    bh, bw = block
    r0 = int(cp.random.randint(0, gh - bh + 1))   # random top-left, block stays in-bounds
    c0 = int(cp.random.randint(0, gw - bw + 1))
    rows = cp.arange(r0, r0 + bh)
    cols = cp.arange(c0, c0 + bw)
    masked_idx  = cp.sort((rows[:, None] * gw + cols[None, :]).ravel())  # flat grid indices
    visible_idx = cp.setdiff1d(cp.arange(gh * gw), masked_idx)
    return tiles[visible_idx], visible_idx, masked_idx

def update_teacher(teacher_modules, student_modules, beta=0.01):
    def _ema(t_arr, s_arr, beta):
        # teacher drifts toward student and away from teacher
        return (1 - beta) * t_arr + beta * s_arr

    def ema_layer(t, s, beta):
        # EMA layers
        t.W = _ema(t.W, s.W, beta)
        t.B = _ema(t.B, s.B, beta)

    def ema_param(t, s, beta):
        # EMA parameters
        t.P = _ema(t.P, s.P, beta)

    def ema_self_attn(t, s, beta):
        # EMA self attention layers
        t.wQ = _ema(t.wQ, s.wQ, beta)
        t.wK = _ema(t.wK, s.wK, beta)
        t.wV = _ema(t.wV, s.wV, beta)

    def ema_mha(t, s, beta):
        # EMA multihead attention with iteration over self attention heads
        t.wO = _ema(t.wO, s.wO, beta)
        for th, sh in zip(t.heads, s.heads):
            ema_self_attn(th, sh, beta)

    def ema_transformer(t, s, beta):
        # Full attention pass
        ema_mha(t.mha, s.mha, beta)
        ema_layer(t.ffn1, s.ffn1, beta)
        ema_layer(t.ffn2, s.ffn2, beta)

    # Extract modules
    ffn1, ffn2, p1, trans1 = teacher_modules
    ffn3, ffn4, p2, trans2 = student_modules

    # Perform EMA based on known structure
    ema_layer(ffn1, ffn3, beta)
    ema_layer(ffn2, ffn4, beta)
    ema_param(p1,  p2,  beta)
    ema_transformer(trans1, trans2, beta)


def collate_masked_with_unmasked(shape, tokens, mask, visible_idx, masked_idx):
    full = cp.zeros(shape, dtype=cp.float32)
    full[visible_idx] = tokens
    full[masked_idx] = mask
    return full