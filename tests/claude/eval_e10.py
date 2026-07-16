"""Sweep every teacher checkpoint in ./Checkpoints; chart representation quality
over training. Pure-NumPy teacher forward (no GPU, reads .npz weights directly).
Run: python tests/claude/eval_e10.py  ->  checkpoint_progression.png + a table."""
import os, re, glob
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, "..", ".."))

def g(W, k): return W[k].astype(np.float64)
def ln(x):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
def dense(W, x, a, b, relu=False):
    y = x @ g(W, a).T + g(W, b); return np.maximum(0, y) if relu else y
def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
def head(W, a, i):
    Q, K, V = a @ g(W, f"head{i}.wQ"), a @ g(W, f"head{i}.wK"), a @ g(W, f"head{i}.wV")
    S = np.einsum("nid,njd->nij", Q, K) / np.sqrt(Q.shape[-1])
    return np.einsum("nij,njd->nid", softmax(S), V)
def trans(W, x):
    x = x + np.concatenate([head(W, ln(x), i) for i in range(4)], -1) @ g(W, "mha.wO")
    b = ln(x)
    return x + dense(W, dense(W, b, "tffn1.W", "tffn1.B", relu=True), "tffn2.W", "tffn2.B")
def encode(W, tiles):
    h = dense(W, tiles, "ffn_a.W", "ffn_a.B", relu=True)
    h = dense(W, h, "ffn_b.W", "ffn_b.B") + g(W, "pos.P")
    return ln(trans(W, h))
def tiles_of(imgs, p=2):
    N, H, Wd = imgs.shape
    return imgs.reshape(N, H//p, p, Wd//p, p).transpose(0, 1, 3, 2, 4).reshape(N, -1, p*p)

X, y = load_digits(return_X_y=True)
T = tiles_of(X.reshape(-1, 8, 8) / 16.0)
tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=0, stratify=y)
ep = lambda p: int(re.search(r"e(\d+)", os.path.basename(p)).group(1))
ckpts = sorted(glob.glob(os.path.join(ROOT, "Checkpoints", "teacher_e*.npz")), key=ep)
print("checkpoints:", [ep(c) for c in ckpts])

R = []
for c in ckpts:
    W = np.load(c)
    reps = encode(W, T); im = reps.mean(1)
    evv = np.linalg.eigvalsh(np.cov((im - im.mean(0)).T))[::-1].clip(min=0)
    eff = evv.sum()**2 / (np.square(evv).sum() + 1e-12)
    xn = reps / (np.linalg.norm(reps, axis=2, keepdims=True) + 1e-8)
    iu = np.triu_indices(16, 1)
    pdiv = (1 - np.einsum("nid,njd->nij", xn, xn)[:, iu[0], iu[1]]).mean()
    km = KMeans(10, n_init=10, random_state=0).fit(im)
    sil = silhouette_score(im, km.labels_)
    Z = StandardScaler().fit(im[tr]).transform(im)
    al = accuracy_score(y[te], LogisticRegression(max_iter=2000).fit(Z[tr], y[tr]).predict(Z[te]))
    ak = accuracy_score(y[te], KNeighborsClassifier(10).fit(Z[tr], y[tr]).predict(Z[te]))
    p2 = PCA(2, random_state=0).fit_transform(im)
    R.append(dict(e=ep(c), eff=eff, pd=pdiv, sil=sil, al=al, ak=ak, p2=p2, km=km.labels_))

print("epoch  lin   knn   effdim  pdiv   sil")
for r in R:
    print(f"{r['e']:>4}  {r['al']:.3f} {r['ak']:.3f}  {r['eff']:5.1f}  {r['pd']:.3f}  {r['sil']:.3f}")

import csv
with open(os.path.join(BASE, "progression.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["epoch","linear","knn","eff_dim","patch_div","silhouette"])
    for r in R:
        w.writerow([r["e"], f"{r['al']:.4f}", f"{r['ak']:.4f}", f"{r['eff']:.3f}", f"{r['pd']:.4f}", f"{r['sil']:.4f}"])
print("saved", os.path.join(BASE, "progression.csv"))

n = len(R)
fig = plt.figure(figsize=(9.5, 2.7*n + 3))
gs = fig.add_gridspec(n+1, 2, height_ratios=[1.3]+[1]*n, hspace=.55, wspace=.2)
e = [r["e"] for r in R]
a0 = fig.add_subplot(gs[0, :])
a0.plot(e, [r["al"] for r in R], "o-", color="#4C72B0", label="linear probe")
a0.plot(e, [r["ak"] for r in R], "s-", color="#55A868", label="kNN(10)")
a0.set_ylim(0, 1); a0.set_xlabel("epoch"); a0.set_ylabel("accuracy")
a0.grid(alpha=.3); a0.legend(loc="lower left", fontsize=8)
a1 = a0.twinx()
a1.plot(e, [r["eff"] for r in R], "^--", color="#C44E52", label="eff dim")
a1.plot(e, [r["pd"] for r in R], "d--", color="#8172B3", label="patch div")
a1.set_ylabel("eff dim / patch div"); a1.legend(loc="lower right", fontsize=8)
a0.set_title("Representation quality across training", fontweight="bold")
s0 = None
for i, r in enumerate(R):
    L = fig.add_subplot(gs[i+1, 0])
    L.scatter(r["p2"][:, 0], r["p2"][:, 1], c=r["km"], cmap="tab10", s=5, alpha=.7)
    L.set_title("e%d  KMeans(10)  sil=%.2f" % (r["e"], r["sil"]), fontsize=9); L.axis("off")
    Rr = fig.add_subplot(gs[i+1, 1])
    s0 = Rr.scatter(r["p2"][:, 0], r["p2"][:, 1], c=y, cmap="tab10", s=5, alpha=.7)
    Rr.set_title("e%d  true digit  probe=%.2f" % (r["e"], r["al"]), fontsize=9); Rr.axis("off")
fig.colorbar(s0, ax=fig.axes[1:], fraction=.015, pad=.01, label="digit", ticks=range(10))
out = os.path.join(BASE, "checkpoint_progression.png")
fig.savefig(out, dpi=130, bbox_inches="tight"); print("saved", out)
