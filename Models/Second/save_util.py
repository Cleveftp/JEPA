import cupy as cp
import numpy as np

# Saving function for teacher courtesy of Claude

def _model_arrays(modules):
    """Yield (name, obj, attr) for every weight array — mirrors the EMA structure.
       modules = (ffn_a, ffn_b, pos, transformer)"""
    ffn_a, ffn_b, pos, trans = modules
    yield "ffn_a.W", ffn_a, "W";  yield "ffn_a.B", ffn_a, "B"
    yield "ffn_b.W", ffn_b, "W";  yield "ffn_b.B", ffn_b, "B"
    yield "pos.P",   pos,   "P"
    yield "mha.wO",  trans.mha, "wO"
    for i, h in enumerate(trans.mha.heads):
        yield f"head{i}.wQ", h, "wQ"
        yield f"head{i}.wK", h, "wK"
        yield f"head{i}.wV", h, "wV"
    yield "tffn1.W", trans.ffn1, "W";  yield "tffn1.B", trans.ffn1, "B"
    yield "tffn2.W", trans.ffn2, "W";  yield "tffn2.B", trans.ffn2, "B"

def save_model(modules, path):
    arrays = {name: cp.asnumpy(getattr(obj, attr))
              for name, obj, attr in _model_arrays(modules)}
    np.savez(path, **arrays)

def load_model(modules, path):
    """
    Simple loading structure 
    ffn1 = Layer(4, 32); ffn2 = Layer(32, token_dim, 'linear')
    p1 = Parameter((stack_dim, token_dim)); trans1 = Transformer(token_dim)
    load_model((ffn1, ffn2, p1, trans1), "teacher.npz")
    """
    if not path.endswith(".npz"):
        path += ".npz"
    data = np.load(path)
    for name, obj, attr in _model_arrays(modules):
        setattr(obj, attr, cp.asarray(data[name])) 