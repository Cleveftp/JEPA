"""Builds a single self-contained interactive HTML: click a patch/token in a digit
and see its attention, nearest-neighbour patches, t-SNE location, and how its
attention evolves across training checkpoints. No GPU, no server -- just open
Visualizations/token_explorer.html in a browser.
Run: python Visualizations/build_token_explorer.py
"""
import os, re, glob, json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
CKDIR = os.path.join(ROOT, "Models/First/Checkpoints")

TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>JEPA token explorer</title>
<style>
 body{margin:0;background:#0f1116;color:#dfe3ea;font:13px system-ui,Segoe UI,Arial}
 h1{font-size:16px;margin:10px 14px;font-weight:600}
 .sub{color:#8b93a3;font-weight:400;font-size:12px}
 #wrap{display:grid;grid-template-columns:150px 300px 1fr;gap:14px;padding:10px 14px}
 .card{background:#171a22;border:1px solid #232838;border-radius:10px;padding:10px}
 .card h2{font-size:12px;margin:0 0 8px;color:#9aa4b8;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
 #gallery{display:grid;grid-template-columns:repeat(6,1fr);gap:3px;max-height:78vh;overflow:auto}
 #gallery canvas{width:100%;image-rendering:pixelated;border-radius:3px;cursor:pointer;border:1px solid #232838}
 #gallery canvas.sel{border-color:#ffd23f;box-shadow:0 0 0 1px #ffd23f}
 canvas{image-rendering:pixelated}
 #right{display:grid;grid-template-columns:1fr 1fr;gap:14px}
 .row{display:flex;gap:6px;flex-wrap:wrap;align-items:flex-start}
 .thumb{display:flex;flex-direction:column;align-items:center;gap:2px;font-size:10px;color:#8b93a3}
 #slider{width:100%}
 .lab{font-size:11px;color:#8b93a3;margin-top:4px}
 b{color:#ffd23f}
</style></head><body>
<h1>JEPA token explorer <span class="sub">click a digit in the gallery, then click a patch (2x2 tile) to inspect its token</span></h1>
<div id="wrap">
  <div class="card"><h2>gallery</h2><div id="gallery"></div></div>
  <div class="card">
    <h2>selected digit</h2>
    <canvas id="digit" width="272" height="272" style="cursor:crosshair"></canvas>
    <div class="lab" id="dlab"></div>
    <h2 style="margin-top:12px">attention from this patch</h2>
    <canvas id="att" width="272" height="272"></canvas>
    <div class="lab">epoch: <b id="eplab"></b></div>
    <input id="slider" type="range" min="0" value="0">
  </div>
  <div id="right">
    <div class="card"><h2>nearest-neighbour patches (cosine)</h2><div class="row" id="nbrs"></div>
      <div class="lab">each: the source digit with the matching patch boxed</div></div>
    <div class="card"><h2>attention over training</h2><div class="row" id="film"></div>
      <div class="lab">this patch's attention map at each checkpoint</div></div>
    <div class="card" style="grid-column:1/3"><h2>token t-SNE (all patch tokens, coloured by digit)</h2>
      <canvas id="tsne" width="720" height="360"></canvas>
      <div class="lab">the <b>white ring</b> is your selected token</div></div>
  </div>
</div>
<script>
const D=__DATA__;
const PAL=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"];
let fr=0, patch=5, epi=D.epochs.length-1;
const S=document.getElementById('slider'); S.max=D.epochs.length-1; S.value=epi;

function px(v){return Math.round(v/16*255);}          // image val 0..16 -> gray
function heat(v){ // 0..1 -> inferno-ish
  v=Math.max(0,Math.min(1,v));
  const r=Math.min(255,60+v*230), g=Math.max(0,v*210-40), b=Math.max(0, v<0.5? 90-v*120 : (v-0.5)*300);
  return 'rgb('+(r|0)+','+(g|0)+','+(b|0)+')';
}
function drawDigit(ctx,idx,cell,hi){
  const im=D.images[idx];
  for(let r=0;r<8;r++)for(let c=0;c<8;c++){const v=px(im[r*8+c]);ctx.fillStyle='rgb('+v+','+v+','+v+')';ctx.fillRect(c*cell,r*cell,cell,cell);}
  if(hi!=null){const pr=(hi/4|0),pc=hi%4;ctx.strokeStyle='#ffd23f';ctx.lineWidth=2;ctx.strokeRect(pc*2*cell+1,pr*2*cell+1,2*cell-2,2*cell-2);}
}
function drawAtt(ctx,idx,a,cell){
  const im=D.images[idx];
  for(let r=0;r<8;r++)for(let c=0;c<8;c++){const v=px(im[r*8+c])*0.35|0;ctx.fillStyle='rgb('+v+','+v+','+v+')';ctx.fillRect(c*cell,r*cell,cell,cell);}
  let mx=Math.max.apply(null,a);
  for(let p=0;p<16;p++){const pr=(p/4|0),pc=p%4;ctx.fillStyle=heat(a[p]/(mx+1e-9));ctx.globalAlpha=0.62;ctx.fillRect(pc*2*cell,pr*2*cell,2*cell,2*cell);}
  ctx.globalAlpha=1;
}
function render(){
  const idx=D.focus[fr];
  drawDigit(document.getElementById('digit').getContext('2d'),idx,34,patch);
  document.getElementById('dlab').innerHTML='label <b>'+D.labels[idx]+'</b> &nbsp; patch <b>'+patch+'</b>';
  const a=D.focusAttn[fr][epi][patch];
  drawAtt(document.getElementById('att').getContext('2d'),idx,a,34);
  document.getElementById('eplab').textContent='e'+D.epochs[epi];
  // neighbours
  const nb=D.neigh[fr*16+patch]; const nd=document.getElementById('nbrs'); nd.innerHTML='';
  nb.forEach(function(t){const d=document.createElement('div');d.className='thumb';
    const cv=document.createElement('canvas');cv.width=cv.height=64;const cx=cv.getContext('2d');
    drawDigit(cx,t[0],8,t[1]); d.appendChild(cv);
    const s=document.createElement('span');s.textContent=D.labels[t[0]]+' / p'+t[1];d.appendChild(s);nd.appendChild(d);});
  // filmstrip
  const fd=document.getElementById('film'); fd.innerHTML='';
  for(let e=0;e<D.epochs.length;e++){const d=document.createElement('div');d.className='thumb';
    const cv=document.createElement('canvas');cv.width=cv.height=52;const cx=cv.getContext('2d');
    const aa=D.focusAttn[fr][e][patch];let mx=Math.max.apply(null,aa);
    for(let p=0;p<16;p++){const pr=(p/4|0),pc=p%4;cx.fillStyle=heat(aa[p]/(mx+1e-9));cx.fillRect(pc*13,pr*13,13,13);}
    if(e===epi){cx.strokeStyle='#ffd23f';cx.lineWidth=3;cx.strokeRect(0,0,52,52);}
    d.appendChild(cv);const s=document.createElement('span');s.textContent='e'+D.epochs[e];d.appendChild(s);fd.appendChild(d);}
  drawTsne();
}
function drawTsne(){
  const cv=document.getElementById('tsne'),cx=cv.getContext('2d');cx.clearRect(0,0,cv.width,cv.height);
  const W=cv.width,H=cv.height,pad=14;
  const X=D.tsne.x,Y=D.tsne.y,L=D.tsne.label;
  for(let i=0;i<X.length;i++){cx.fillStyle=PAL[L[i]];cx.globalAlpha=0.55;
    cx.beginPath();cx.arc(pad+X[i]*(W-2*pad),pad+Y[i]*(H-2*pad),2.2,0,7);cx.fill();}
  cx.globalAlpha=1;
  const tid=fr*16+patch; // focus tokens are first, in focus-major order
  const sx=pad+X[tid]*(W-2*pad),sy=pad+Y[tid]*(H-2*pad);
  cx.strokeStyle='#fff';cx.lineWidth=2.5;cx.beginPath();cx.arc(sx,sy,7,0,7);cx.stroke();
  cx.fillStyle=PAL[L[tid]];cx.beginPath();cx.arc(sx,sy,4,0,7);cx.fill();
}
// gallery
const gal=document.getElementById('gallery');
D.focus.forEach(function(idx,i){const cv=document.createElement('canvas');cv.width=cv.height=32;
  drawDigit(cv.getContext('2d'),idx,4,null);cv.dataset.i=i;
  cv.onclick=function(){fr=i;[...gal.children].forEach(c=>c.classList.remove('sel'));cv.classList.add('sel');render();};
  gal.appendChild(cv);});
gal.children[0].classList.add('sel');
document.getElementById('digit').onclick=function(ev){const b=this.getBoundingClientRect();
  const c=Math.floor((ev.clientX-b.left)/34),r=Math.floor((ev.clientY-b.top)/34);
  patch=(Math.floor(r/2))*4+Math.floor(c/2);render();};
S.oninput=function(){epi=+S.value;render();};
render();
</script></body></html>"""

def g(W,k): return W[k].astype(np.float64)
def ln(x): return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)
def dn(W,x,a,b,r=False):
    y=x@g(W,a).T+g(W,b); return np.maximum(0,y) if r else y
def sm(s):
    e=np.exp(s-s.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)
def tiles_of(imgs,p=2):
    N,H,Wd=imgs.shape
    return imgs.reshape(N,H//p,p,Wd//p,p).transpose(0,1,3,2,4).reshape(N,-1,p*p)
def fwd(W,T,attn=False):
    h=dn(W,T,"ffn_a.W","ffn_a.B",True); h=dn(W,h,"ffn_b.W","ffn_b.B")+g(W,"pos.P")
    a=ln(h); outs=[]; A4=[]
    for i in range(4):
        Q,K,V=a@g(W,f"head{i}.wQ"),a@g(W,f"head{i}.wK"),a@g(W,f"head{i}.wV")
        A=sm(np.einsum("nid,njd->nij",Q,K)/np.sqrt(Q.shape[-1])); A4.append(A)
        outs.append(np.einsum("nij,njd->nid",A,V))
    x=h+np.concatenate(outs,-1)@g(W,"mha.wO"); b=ln(x)
    rep=ln(x+dn(W,dn(W,b,"tffn1.W","tffn1.B",True),"tffn2.W","tffn2.B"))
    return (rep, np.stack(A4,1)) if attn else rep

X,y=load_digits(return_X_y=True); N=len(y)
imgs=X.reshape(-1,8,8)/16.0
T=tiles_of(imgs)
ck=sorted(glob.glob(os.path.join(CKDIR,"teacher_e*.npz")),
          key=lambda p:int(re.search(r"e(\d+)",p).group(1)))
epochs=[int(re.search(r"e(\d+)",c).group(1)) for c in ck]
Ws=[np.load(c) for c in ck]; Wlast=Ws[-1]
print("checkpoints", epochs)

focus=np.concatenate([np.where(y==d)[0][:6] for d in range(10)]).tolist()
nf=len(focus)
Tf=T[focus]
attn_by_ep=[fwd(W,Tf,attn=True)[1].mean(1) for W in Ws]
focusAttn=[[np.round(attn_by_ep[e][fr],3).tolist() for e in range(len(Ws))] for fr in range(nf)]

tok=fwd(Wlast,T)
flat=tok.reshape(N*16,64)
fn=flat/(np.linalg.norm(flat,axis=1,keepdims=True)+1e-9)
img_of=np.repeat(np.arange(N),16); patch_of=np.tile(np.arange(16),N)

focus_tok_ids=np.array([fi*16+p for fi in focus for p in range(16)])
sims=fn[focus_tok_ids]@fn.T
neigh=[]
for r,tid in enumerate(focus_tok_ids):
    order=np.argsort(-sims[r]); src=img_of[tid]; out=[]
    for j in order:
        if img_of[j]==src: continue
        out.append([int(img_of[j]),int(patch_of[j])])
        if len(out)==6: break
    neigh.append(out)

extra=np.random.RandomState(0).choice(N*16,1200,replace=False)
bank=np.concatenate([focus_tok_ids,extra])
emb=TSNE(2,perplexity=30,init="pca",random_state=0).fit_transform(flat[bank])
emb=(emb-emb.min(0))/(np.ptp(emb,axis=0)+1e-9)
tsne=dict(x=np.round(emb[:,0],4).tolist(), y=np.round(emb[:,1],4).tolist(),
          label=[int(y[img_of[t]]) for t in bank])

data=dict(epochs=epochs, focus=[int(f) for f in focus], nf=nf,
    images=[[int(v*16) for v in im.ravel()] for im in imgs],
    labels=[int(v) for v in y], focusAttn=focusAttn, neigh=neigh, tsne=tsne)
html=TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",",":")))
out=os.path.join(BASE,"token_explorer.html")
open(out,"w").write(html)
print("saved", out, f"({len(html)//1024} KB)")
