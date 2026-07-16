"""
Representation explorer for the JEPA teacher.  Pure NumPy + matplotlib (no GPU).
Produces several "what does the model see" figures from a teacher checkpoint:

  1) attention_explorer.png -- for one digit, what each patch attends to (+ per-head)
  2) occlusion_saliency.png -- blank each patch, measure how much the rep changes
  3) tsne_galaxy.png        -- t-SNE of all reps, colored by digit, with thumbnails
  4) neighbors.png          -- nearest neighbours in representation space

Run:  python Visualizations/representation_explorer.py [--checkpoint PATH] [--index N]
Add   --interactive  to click patches live (needs a display).
"""
import os, sys, argparse
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib
if "--interactive" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=os.path.join(ROOT, "Models/First/Checkpoints/teacher_e100.npz"))
ap.add_argument("--index", type=int, default=-1, help="dataset index of the focus digit")
ap.add_argument("--interactive", action="store_true")
args = ap.parse_args()

W = np.load(args.checkpoint)
def g(k): return W[k].astype(np.float64)
def ln(x): return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
def dn(x, a, b, r=False):
    y = x @ g(a).T + g(b); return np.maximum(0, y) if r else y
def sm(s):
    e = np.exp(s - s.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def tiles_of(imgs, p=2):
    N, H, Wd = imgs.shape
    return imgs.reshape(N, H//p, p, Wd//p, p).transpose(0, 1, 3, 2, 4).reshape(N, -1, p*p)

def forward(tiles, want_attn=False):
    """tiles: (16,4) -> rep (16,64); optionally per-head attention (4,16,16)."""
    h = dn(tiles, "ffn_a.W", "ffn_a.B", True)
    h = dn(h, "ffn_b.W", "ffn_b.B") + g("pos.P")
    a = ln(h); outs = []; attn = []
    for i in range(4):
        Q, K, V = a @ g(f"head{i}.wQ"), a @ g(f"head{i}.wK"), a @ g(f"head{i}.wV")
        A = sm(Q @ K.T / np.sqrt(Q.shape[-1])); attn.append(A); outs.append(A @ V)
    x = h + np.concatenate(outs, -1) @ g("mha.wO")
    b = ln(x)
    rep = ln(x + dn(dn(b, "tffn1.W", "tffn1.B", True), "tffn2.W", "tffn2.B"))
    return (rep, np.stack(attn)) if want_attn else rep

def encode_all(imgs):
    T = tiles_of(imgs)
    h = dn(T, "ffn_a.W", "ffn_a.B", True)
    h = dn(h, "ffn_b.W", "ffn_b.B") + g("pos.P")
    a = ln(h)
    outs = []
    for i in range(4):
        Q, K, V = a @ g(f"head{i}.wQ"), a @ g(f"head{i}.wK"), a @ g(f"head{i}.wV")
        A = sm(np.einsum("nid,njd->nij", Q, K) / np.sqrt(Q.shape[-1]))
        outs.append(np.einsum("nij,njd->nid", A, V))
    x = h + np.concatenate(outs, -1) @ g("mha.wO")
    b = ln(x)
    return ln(x + dn(dn(b, "tffn1.W", "tffn1.B", True), "tffn2.W", "tffn2.B"))

up = lambda m: np.kron(m, np.ones((2, 2)))          # 4x4 patch grid -> 8x8 pixels

# ---------------- data ----------------
X, y = load_digits(return_X_y=True)
imgs = X.reshape(-1, 8, 8) / 16.0
idx = args.index if args.index >= 0 else int(np.where(y == 3)[0][1])   # a nice '3'
img = imgs[idx]
tiles = tiles_of(img[None])[0]
rep, attn = forward(tiles, want_attn=True)     # attn (4,16,16)
Amean = attn.mean(0)                            # (16,16) mean over heads

def overlay(ax, base, heat, title, cmap="inferno"):
    ax.imshow(base, cmap="gray", interpolation="nearest")
    ax.imshow(up(heat.reshape(4, 4)), cmap=cmap, alpha=0.55, interpolation="bilinear")
    ax.set_title(title, fontsize=8); ax.axis("off")

# ============ 1) ATTENTION EXPLORER ============
fig = plt.figure(figsize=(11, 6.2))
gs = fig.add_gridspec(4, 8, hspace=.35, wspace=.15)
# left: the digit with its 4x4 patch grid
axd = fig.add_subplot(gs[0:2, 0:2])
axd.imshow(img, cmap="gray", interpolation="nearest")
for t in (1.5, 3.5, 5.5): axd.axvline(t, color="cyan", lw=.7); axd.axhline(t, color="cyan", lw=.7)
axd.set_title(f"digit (label {y[idx]})\n16 patches", fontsize=9); axd.axis("off")
# 16 per-patch attention maps, laid out in patch position
for q in range(16):
    r, c = divmod(q, 4)
    ax = fig.add_subplot(gs[r, 2 + c])
    overlay(ax, img, Amean[q], f"patch {q}")
# bottom-left block: the 4 heads for the center patch
qc = 5
for hd in range(4):
    ax = fig.add_subplot(gs[2 + hd // 2, hd % 2])
    overlay(ax, img, attn[hd, qc], f"head {hd} (patch {qc})", cmap="viridis")
fig.suptitle("Attention: what each patch looks at  (mean over heads, per patch)", fontweight="bold")
fig.savefig(os.path.join(BASE, "attention_explorer.png"), dpi=130, bbox_inches="tight")
print("saved attention_explorer.png")

# ============ 2) OCCLUSION SALIENCY ============
def saliency(im):
    t = tiles_of(im[None])[0]
    r0 = forward(t).mean(0)
    s = np.zeros(16)
    for k in range(16):
        t2 = t.copy(); t2[k] = 0.0
        s[k] = np.linalg.norm(r0 - forward(t2).mean(0))
    return s
reps_all = encode_all(imgs).mean(1)
fig, axs = plt.subplots(2, 10, figsize=(15, 3.4))
for d in range(10):
    j = int(np.where(y == d)[0][0])
    axs[0, d].imshow(imgs[j], cmap="gray"); axs[0, d].set_title(str(d), fontsize=9); axs[0, d].axis("off")
    s = saliency(imgs[j]); s = (s - s.min()) / (np.ptp(s) + 1e-9)
    overlay(axs[1, d], imgs[j], s, "")
axs[0, 0].set_ylabel("digit"); axs[1, 0].set_ylabel("saliency")
fig.suptitle("Occlusion saliency: which patches most change the representation when removed", fontweight="bold")
fig.savefig(os.path.join(BASE, "occlusion_saliency.png"), dpi=130, bbox_inches="tight")
print("saved occlusion_saliency.png")

# ============ 3) t-SNE GALAXY ============
sub = np.random.RandomState(0).choice(len(y), 700, replace=False)
emb = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(reps_all[sub])
fig, ax = plt.subplots(figsize=(9, 8))
sc = ax.scatter(emb[:, 0], emb[:, 1], c=y[sub], cmap="tab10", s=14, alpha=.65)
for k in np.random.RandomState(1).choice(len(sub), 90, replace=False):
    im = OffsetImage(imgs[sub[k]], cmap="gray", zoom=1.1)
    ax.add_artist(AnnotationBbox(im, emb[k], frameon=False))
ax.set_title("t-SNE of JEPA representations (700 digits) with thumbnails", fontweight="bold")
ax.axis("off"); fig.colorbar(sc, label="digit", ticks=range(10))
fig.savefig(os.path.join(BASE, "tsne_galaxy.png"), dpi=130, bbox_inches="tight")
print("saved tsne_galaxy.png")

# ============ 4) NEAREST NEIGHBOURS ============
Rn = reps_all / (np.linalg.norm(reps_all, axis=1, keepdims=True) + 1e-9)
queries = [int(np.where(y == d)[0][0]) for d in (0, 3, 5, 8, 9, 6)]
fig, axs = plt.subplots(len(queries), 6, figsize=(7, 1.15 * len(queries)))
for row, q in enumerate(queries):
    sims = Rn @ Rn[q]; order = np.argsort(-sims)
    order = order[order != q][:5]
    axs[row, 0].imshow(imgs[q], cmap="gray"); axs[row, 0].axis("off")
    axs[row, 0].set_title("query", fontsize=8)
    for j, nb in enumerate(order):
        axs[row, j + 1].imshow(imgs[nb], cmap="gray"); axs[row, j + 1].axis("off")
        axs[row, j + 1].set_title(f"{sims[nb]:.2f}", fontsize=8)
fig.suptitle("Nearest neighbours in representation space (cosine)", fontweight="bold")
fig.savefig(os.path.join(BASE, "neighbors.png"), dpi=130, bbox_inches="tight")
print("saved neighbors.png")

# ============ optional: interactive attention ============
if args.interactive:
    figi, (aL, aR) = plt.subplots(1, 2, figsize=(8, 4))
    aL.imshow(img, cmap="gray"); aL.set_title("click a patch"); aL.axis("off")
    for t in (1.5, 3.5, 5.5): aL.axvline(t, color="cyan", lw=.7); aL.axhline(t, color="cyan", lw=.7)
    overlay(aR, img, Amean[qc], "attention")
    def onclick(ev):
        if ev.inaxes == aL and ev.xdata is not None:
            q = int(ev.ydata // 2) * 4 + int(ev.xdata // 2)
            aR.clear(); overlay(aR, img, Amean[q], f"patch {q} attends to"); figi.canvas.draw()
    figi.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
