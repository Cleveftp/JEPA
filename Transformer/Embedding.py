import cupy as cp
import hashlib
from sklearn.datasets import load_digits

class Embedding:
    def __init__(self, dim=128, tokens=None):
        self.dim = dim

        if tokens:
            self.tokens = tokens
        else:
            self.tokens = {}

    def embed(self, token, value):
        assert value.shape[0] == self.dim
    
        self.tokens[token] = value

    def hash_embed(self, region, value):
        token = self._hash_region(region)
        self.embed(token, value)

    def get_value(self, token):
        if token in self.tokens.keys():
            return self.tokens[token]
        
    def _hash_region(self, region):
        # Region -> Hash
        q = region.astype(cp.float32)
        return hashlib.sha256(q.get().tobytes()).digest()
    
    def _tile(self, image, p=2):
        # Splits the image into regions
        H, W = image.shape[:2]
        return (image.reshape(H // p, p, W // p, p)         # (4,2,4,2)
                    .transpose(0, 2, 1, 3)                 # (4,4,2,2)
                    .reshape(-1, p, p))                    # (16, 2, 2)
    
    def visualize(self, method="pca", annotate=False):
        # Visualization brought to you by claude

        import numpy as np
        import matplotlib.pyplot as plt

        if len(self.tokens) < 2:
            print("need at least 2 tokens to visualize")
            return

        keys = list(self.tokens.keys())
        M = cp.stack([self.tokens[k].ravel() for k in keys])   # (N, dim) on GPU
        M = cp.asnumpy(M).astype(np.float64)                   # to CPU for plotting
        M -= M.mean(axis=0, keepdims=True)                     # center

        if method == "pca":
            # PCA via SVD: project onto the top-2 principal directions
            U, S, _ = np.linalg.svd(M, full_matrices=False)
            coords = U[:, :2] * S[:2]
        elif method == "tsne":
            from sklearn.manifold import TSNE
            coords = TSNE(n_components=2,
                        perplexity=min(30, len(keys) - 1)).fit_transform(M)
        else:
            raise ValueError("method must be 'pca' or 'tsne'")

        plt.figure(figsize=(7, 7))
        plt.scatter(coords[:, 0], coords[:, 1], s=20, alpha=0.7)
        if annotate:
            for (x, y), k in zip(coords, keys):
                plt.annotate(k[:3].hex(), (x, y), fontsize=6)   # short hash tag
        plt.title(f"Embedding space — {len(keys)} tokens ({method.upper()})")
        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":
    embed_space = Embedding()

    X, _ = load_digits(return_X_y=True)
    X = cp.expand_dims(cp.asarray(X).reshape(-1, 8, 8), axis=1) / 16

    tiles = embed_space._tile(X[0, 0])
    o1 = embed_space._hash_region(tiles[0])
    o2 = embed_space._hash_region(tiles[-1])

    print(o1 == o2)


