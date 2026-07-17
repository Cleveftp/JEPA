import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.abspath(__file__))          # tests/claude (my folder)
ROOT = os.path.abspath(os.path.join(BASE, "..", ".."))     # JEPA project root

tr = np.load(os.path.join(ROOT, "features.npy"))   # (N,16,64)
N = tr.shape[0]
print(f"train {tr.shape}")

# --- image-level rep = mean over patches ---
img = tr.mean(axis=1)            # (N,64)

# --- basic health ---
patch = tr.reshape(-1, 64)
print("\n== per-patch stats (teacher output, LN'd) ==")
print(f"  patch L2 norm: mean {np.linalg.norm(patch,axis=1).mean():.3f}  std {np.linalg.norm(patch,axis=1).std():.3f}")
dead = (patch.std(axis=0) < 1e-4).sum()
print(f"  dead dims (std<1e-4): {dead}/64")

# --- dimensional collapse: participation ratio on image reps ---
Xc = img - img.mean(0)
cov = np.cov(Xc.T)
ev = np.linalg.eigvalsh(cov)[::-1].clip(min=0)
pr = (ev.sum()**2) / (np.square(ev).sum() + 1e-12)   # effective dimensionality
cum = np.cumsum(ev)/ev.sum()
d90 = int(np.searchsorted(cum, 0.90))+1
d95 = int(np.searchsorted(cum, 0.95))+1
print("\n== dimensional collapse (image reps, 64-d) ==")
print(f"  participation ratio (effective dim): {pr:.1f} / 64")
print(f"  dims for 90% var: {d90}   95% var: {d95}")

# --- patch diversity within an image (spatial collapse check) ---
def within_img_div(x):
    # mean pairwise cosine DISTANCE among the 16 patches, averaged over images
    xn = x / (np.linalg.norm(x, axis=2, keepdims=True)+1e-8)
    sims = np.einsum("nid,njd->nij", xn, xn)
    iu = np.triu_indices(x.shape[1], k=1)
    return (1 - sims[:, iu[0], iu[1]]).mean()
print("\n== patch diversity within images ==")
print(f"  mean pairwise cosine distance between patches: {within_img_div(tr):.3f}  (0=all identical, ~1=orthogonal)")

# --- unsupervised cluster structure at k=10 (digits) ---
km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(img)
sil = silhouette_score(img, km.labels_)
sizes = np.bincount(km.labels_, minlength=10)
print("\n== KMeans(10) on image reps ==")
print(f"  silhouette: {sil:.3f}")
print(f"  cluster sizes: {sorted(sizes.tolist(), reverse=True)}  (balanced target ~{N//10})")

# --- viz: PCA(2) colored by cluster ---
p2 = PCA(2, random_state=0).fit_transform(img)
plt.figure(figsize=(7,6))
sc = plt.scatter(p2[:,0], p2[:,1], c=km.labels_, cmap="tab10", s=8, alpha=0.7)
plt.title(f"e10 teacher reps — PCA(2), KMeans(10)\nsilhouette={sil:.3f}  eff.dim={pr:.1f}/64")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.colorbar(sc, label="cluster")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "rep_quality_e10.png"), dpi=130)
print("\nsaved rep_quality_e10.png")
