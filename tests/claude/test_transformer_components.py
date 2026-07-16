"""
Component tests for the Transformer stack.

Run directly:      python tests/claude/test_transformer_components.py
Or with pytest:    pytest tests/claude/test_transformer_components.py

Strategy
--------
Your layers apply the weight update *inside* backward (`self.W -= lr*dW`) and
don't expose the raw gradient. So the analytic gradient is recovered by:
    snapshot W  ->  backward(dO, lr)  ->  analytic = (W_before - W_after)/lr
That analytic value is then compared against a numerical finite-difference
gradient computed from forward-only passes. A transpose/sign bug shows up as a
relative error near 1.0; a correct gradient sits well under the tolerance.

Composite modules (multi-head, transformer block) are checked with shape tests
plus an "it actually learns" test: if backprop is wired correctly the MSE to a
fixed random target drops sharply.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cupy as cp

from NeuralNetwork.neural_network import Layer
from Transformer.self_attention import self_attn
from Transformer.multi_head_attn import multi_head_attention

# The block class name is spelled "Transfomer" in transformer.py; accept either.
try:
    from Transformer.transformer import Transfomer as TransformerBlock
except ImportError:  # in case the typo gets fixed later
    from Transformer.transformer import Transformer as TransformerBlock

from Transformer.Embedding import Embedding


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _mse(out, target):
    """Loss (float64 for a clean finite difference) and its gradient dO."""
    diff = out - target
    loss = float(cp.mean(diff.astype(cp.float64) ** 2))
    dO = (2.0 * diff / out.size).astype(out.dtype)
    return loss, dO


def _rel_err(a, b):
    # Standard gradient-check metric: aggregate norm ratio, not per-element max.
    # A per-element max is ill-conditioned wherever the true gradient is ~0
    # (float32 finite-difference noise dominates), which throws false failures.
    a = cp.asarray(a, cp.float64).ravel()
    b = cp.asarray(b, cp.float64).ravel()
    denom = float(cp.linalg.norm(a) + cp.linalg.norm(b))
    return 0.0 if denom == 0.0 else float(cp.linalg.norm(a - b) / denom)


def _numeric_grad(forward_loss, arr, eps=1e-3):
    """Central finite-difference gradient of a scalar loss wrt every element of
    `arr` (modified in place through its ravelled view, then restored)."""
    grad = cp.zeros_like(arr)
    flat = arr.ravel()
    gflat = grad.ravel()
    for i in range(flat.size):
        original = flat[i].copy()
        flat[i] = original + eps
        lp = forward_loss()
        flat[i] = original - eps
        lm = forward_loss()
        flat[i] = original
        gflat[i] = (lp - lm) / (2 * eps)
    return grad


def _check_param_grads(module, x, target, names, lr=1.0, tol=2e-2):
    """Gradient-check the named weight attributes of `module`.
    Returns the analytic dX from backward so the caller can also check it."""
    # analytic weight grads via the update-extraction trick
    out = module.forward(x)
    _, dO = _mse(out, target)
    before = {n: getattr(module, n).copy() for n in names}
    dX = module.backward(dO, lr)
    analytic = {n: (before[n] - getattr(module, n)) / lr for n in names}
    for n in names:  # restore weights
        setattr(module, n, before[n].copy())

    # numeric weight grads
    for n in names:
        W = getattr(module, n)
        num = _numeric_grad(lambda: _mse(module.forward(x), target)[0], W)
        err = _rel_err(analytic[n], num)
        assert err < tol, (
            f"{module.__class__.__name__}.{n} gradient mismatch: rel_err={err:.4e}"
        )
    return dX


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def test_embedding_tile_shape():
    emb = Embedding(dim=4)
    img = cp.arange(64, dtype=cp.float32).reshape(8, 8)
    tiles = emb._tile(img, p=2)
    assert tiles.shape == (16, 2, 2), f"tile shape wrong: {tiles.shape}"


def test_embedding_hash_is_deterministic():
    emb = Embedding(dim=4)
    region = cp.random.randn(2, 2).astype(cp.float32)
    assert emb._hash_region(region) == emb._hash_region(region), "hash not deterministic"


def test_embedding_hash_distinguishes_regions():
    emb = Embedding(dim=4)
    a = cp.zeros((2, 2), cp.float32)
    b = cp.ones((2, 2), cp.float32)
    assert emb._hash_region(a) != emb._hash_region(b), "distinct regions collided"


def test_embedding_store_and_get():
    emb = Embedding(dim=4)
    tok = emb._hash_region(cp.ones((2, 2), cp.float32))
    vec = cp.arange(4, dtype=cp.float32)
    emb.tokens[tok] = vec  # (embed() has a `.shape()` bug — store directly for now)
    assert cp.allclose(emb.get_value(tok), vec), "get_value did not round-trip"
    assert emb.get_value(b"missing") is None, "unknown token should return None"


# --------------------------------------------------------------------------- #
# Layer in sequence (2-D) mode  — the change you made to support (N, dim)
# --------------------------------------------------------------------------- #
def test_layer_2d_forward_shape():
    layer = Layer(6, 5, "linear")
    x = cp.random.randn(4, 6).astype(cp.float32)
    assert layer.forward(x).shape == (4, 5), "Layer 2-D forward shape wrong"


def test_layer_2d_weight_grads():
    cp.random.seed(0)
    layer = Layer(6, 5, "linear")
    x = cp.random.randn(4, 6).astype(cp.float32)
    target = cp.random.randn(4, 5).astype(cp.float32)
    dX = _check_param_grads(layer, x, target, ["W", "B"])
    assert dX.shape == x.shape, "Layer dX shape wrong"


# --------------------------------------------------------------------------- #
# Single-head self attention
# --------------------------------------------------------------------------- #
def test_self_attn_shapes():
    N, dim = 5, 8
    attn = self_attn(input_dim=dim, head_dim=dim)
    out = attn.forward(cp.random.randn(N, dim).astype(cp.float32))
    assert out.shape == (N, dim), f"self_attn output shape wrong: {out.shape}"


def test_self_attn_weight_grads():
    cp.random.seed(1)
    N, dim = 4, 6
    attn = self_attn(input_dim=dim, head_dim=dim)
    x = cp.random.randn(N, dim).astype(cp.float32)
    target = cp.random.randn(N, dim).astype(cp.float32)
    _check_param_grads(attn, x, target, ["wQ", "wK", "wV"])


def test_self_attn_input_grad():
    cp.random.seed(2)
    N, dim = 4, 6
    attn = self_attn(input_dim=dim, head_dim=dim)
    x = cp.random.randn(N, dim).astype(cp.float32)
    target = cp.random.randn(N, dim).astype(cp.float32)

    out = attn.forward(x)
    _, dO = _mse(out, target)
    before = {n: getattr(attn, n).copy() for n in ("wQ", "wK", "wV")}
    dX = attn.backward(dO, 1.0)
    for n in before:  # restore so the numeric check sees original weights
        setattr(attn, n, before[n])

    num = _numeric_grad(lambda: _mse(attn.forward(x), target)[0], x)
    err = _rel_err(dX, num)
    assert err < 2e-2, f"self_attn dX mismatch: rel_err={err:.4e}"


# --------------------------------------------------------------------------- #
# Multi-head attention
# --------------------------------------------------------------------------- #
def test_multihead_shape():
    N, dim = 6, 8
    mha = multi_head_attention(dim, n_heads=2)
    out = mha.forward(cp.random.randn(N, dim).astype(cp.float32))
    assert out.shape == (N, dim), f"multihead output shape wrong: {out.shape}"


def test_multihead_learns():
    cp.random.seed(3)
    N, dim = 4, 8
    x = cp.random.randn(N, dim).astype(cp.float32)
    target = cp.random.randn(N, dim).astype(cp.float32)
    mha = multi_head_attention(dim, n_heads=2)

    first = None
    for i in range(400):
        out = mha.forward(x)
        loss, dO = _mse(out, target)
        if i == 0:
            first = loss
        mha.backward(dO, 0.05)
    assert loss < first * 0.3, f"multihead failed to learn: {first:.4f} -> {loss:.4f}"


# --------------------------------------------------------------------------- #
# Transformer block  (expects the residual/FFN fix + dimension preservation)
# --------------------------------------------------------------------------- #
def test_transformer_preserves_shape():
    N, dim = 5, 8
    block = TransformerBlock(dim, dim, n_heads=2)
    x = cp.random.randn(N, dim).astype(cp.float32)
    out = block.forward(x)
    assert out.shape == x.shape, (
        f"transformer must preserve shape for residuals, got {out.shape} vs {x.shape}"
    )


def test_transformer_learns():
    cp.random.seed(4)
    N, dim = 4, 8
    x = cp.random.randn(N, dim).astype(cp.float32)
    target = cp.random.randn(N, dim).astype(cp.float32)
    block = TransformerBlock(dim, dim, n_heads=2)

    first = None
    for i in range(400):
        out = block.forward(x)
        loss, dO = _mse(out, target)
        if i == 0:
            first = loss
        block.backward(dO, 0.02)
    assert loss < first * 0.5, f"transformer failed to learn: {first:.4f} -> {loss:.4f}"


# --------------------------------------------------------------------------- #
# simple runner (no pytest needed)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001 - want to keep going and report all
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
