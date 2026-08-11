import sys, json, numpy as np, scipy.sparse as sp
sys.path.insert(0,"/home/jrich/Desktop/rgit-private")
import anndata as ad
from scipy.stats import rankdata
from scipy.special import ndtri
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
REPO="/home/jrich/Desktop/rgit-private/"
SD="/home/jrich/tmp/claude-1023/-home-jrich-Desktop-rgit/40fecd3b-7b56-47e4-9518-f88caeedf0bc/scratchpad/"
dn=lambda M: M.toarray() if sp.issparse(M) else np.asarray(M)
def grt(M):
    M=np.asarray(M,float);n=M.shape[0]
    return ndtri(np.apply_along_axis(lambda c: rankdata(c,method="average"),0,M)/(n+1.0))
def untied(M,mt=0.5):
    return np.array([np.unique(M[:,j],return_counts=True)[1].max()/M.shape[0]<=mt for j in range(M.shape[1])])
def work(M,k,seed=0):
    Z=StandardScaler().fit_transform(grt(M));Z=Z[:,np.isfinite(Z).all(0)]
    return PCA(min(k,Z.shape[1],Z.shape[0]-1),random_state=seed).fit_transform(Z)
resid=lambda M,D: M-D@np.linalg.lstsq(D,M,rcond=None)[0]

g=ad.read_h5ad(REPO+"data/tcga_kirc/genomics/gene_expression.h5ad")
sym=np.asarray(g.var["gene_name"])
Gl=dn(g.layers["tpm_unstranded"]).astype(np.float64)
libs=Gl.sum(1,keepdims=True);libs[libs==0]=1.0
Gl=np.log1p(Gl/libs*np.median(libs))
ok=np.isfinite(Gl).all(0)&((Gl>0).mean(0)>0.1);Gl=Gl[:,ok];sym=sym[ok]
k=untied(Gl);Gl=Gl[:,k];sym=sym[k]
pids=list(g.obs_names);n=len(pids);PS=n//5
hv=np.argsort(Gl.var(0))[::-1][:2000]
Y=StandardScaler().fit_transform(grt(Gl[:,hv])); ynames=sym[hv]
D_demo=np.load(SD+"D_demo.npy")
kf=KFold(5,shuffle=True,random_state=0); ALPHAS=np.logspace(-1,5,25)

def oof_r2(X,Y):
    """Per-gene out-of-fold R^2, alpha chosen by mean out-of-fold R^2."""
    best=None
    for al in ALPHAS:
        P=np.zeros_like(Y)
        for tr,te in kf.split(X):
            P[te]=Ridge(alpha=al).fit(X[tr],Y[tr]).predict(X[te])
        r2=1-((Y-P)**2).sum(0)/((Y-Y.mean(0))**2).sum(0)
        if best is None or r2.mean()>best[1]: best=(al,r2.mean(),r2)
    return best

OUT={}
for IMG in ["tumor_radiomics","organ_radiomics","tumor_radimagenet","whole_radimagenet"]:
    a=ad.read_h5ad(REPO+f"data/tcga_kirc/imaging/{IMG}.h5ad")[pids]
    Xa=dn(a.X).astype(np.float64);Xa=Xa[:,Xa.std(0)>0];Xa=Xa[:,untied(Xa)]
    Xw=work(Xa,PS)
    al,mean_r2,r2=oof_r2(Xw,Y)
    # permutation null on the same aggregate
    rng=np.random.default_rng(0); nulls=[]
    for b in range(50):
        P=np.zeros_like(Y); Xp=Xw[rng.permutation(n)]
        for tr,te in kf.split(Xp):
            P[te]=Ridge(alpha=al).fit(Xp[tr],Y[tr]).predict(Xp[te])
        nulls.append((1-((Y-P)**2).sum(0)/((Y-Y.mean(0))**2).sum(0)).mean())
    nulls=np.array(nulls)
    ald,mean_d,r2d=oof_r2(resid(Xw,D_demo),resid(Y,D_demo))
    top=np.argsort(r2)[::-1][:10]
    OUT[IMG]=dict(alpha=float(al),mean_r2=float(mean_r2),median_r2=float(np.median(r2)),
                  max_r2=float(r2.max()),best_gene=str(ynames[r2.argmax()]),
                  frac_pos=float((r2>0).mean()),
                  null_mean=float(nulls.mean()),null_q95=float(np.quantile(nulls,.95)),
                  p=float((1+np.sum(nulls>=mean_r2))/(1+len(nulls))),
                  mean_r2_deconf=float(mean_d),
                  top_genes=[{"gene":str(ynames[j]),"r2":float(r2[j])} for j in top])
    o=OUT[IMG]
    print(f"[{IMG:20s}] mean per-gene CV R2={o['mean_r2']:+.4f} (null {o['null_mean']:+.4f}, "
          f"q95 {o['null_q95']:+.4f}, p={o['p']:.3f}) | median={o['median_r2']:+.4f} "
          f"max={o['max_r2']:.3f} ({o['best_gene']}) | frac>0={o['frac_pos']:.2f} | -demo {o['mean_r2_deconf']:+.4f}")
    tops=", ".join("%s(%.2f)"%(t["gene"],t["r2"]) for t in o["top_genes"][:6])
    print("    top: "+tops)
json.dump(OUT,open(SD+"budget.json","w"),indent=2)
print("wrote budget.json")
