import sys, json, numpy as np, scipy.sparse as sp
sys.path.insert(0,"/home/jrich/Desktop/rgit-private")
import anndata as ad
from scipy.stats import rankdata
from scipy.special import ndtri
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rgit import fit_recoverability, cross_validated_recoverability, permutation_test
REPO="/home/jrich/Desktop/rgit-private/"
dn=lambda M: M.toarray() if sp.issparse(M) else np.asarray(M)
def grt(M):
    M=np.asarray(M,float);n=M.shape[0]
    return ndtri(np.apply_along_axis(lambda c: rankdata(c,method="average"),0,M)/(n+1.0))
def untied(M,mt=0.5):
    return np.array([np.unique(M[:,j],return_counts=True)[1].max()/M.shape[0]<=mt
                     for j in range(M.shape[1])])
def work(M,k,seed=0):
    Z=StandardScaler().fit_transform(grt(M));Z=Z[:,np.isfinite(Z).all(0)]
    return PCA(min(k,Z.shape[1],Z.shape[0]-1),random_state=seed).fit_transform(Z)

g=ad.read_h5ad(REPO+"data/tcga_kirc/genomics/gene_expression.h5ad")
Gl=dn(g.layers["tpm_unstranded"]).astype(np.float64)
libs=Gl.sum(1,keepdims=True);libs[libs==0]=1.0
Gl=np.log1p(Gl/libs*np.median(libs))
ok=np.isfinite(Gl).all(0)&((Gl>0).mean(0)>0.1);Gl=Gl[:,ok];Gl=Gl[:,untied(Gl)]
pids=list(g.obs_names);n=len(pids);PS=n//5
hv=np.argsort(Gl.var(0))[::-1][:2000]
Gw=work(Gl[:,hv],PS)

def stat_in(G,X):
    return float(fit_recoverability(G,X,n_components=1).recoverability[0])
def stat_cv(G,X,seed=0):
    return float(cross_validated_recoverability(G,X,n_components=1,n_folds=5,
                 random_state=seed).mean(0)[0])

def simulate(nn,p,d,rho2,sd_g,sd_x,B,rng):
    """B draws with ONE planted population canonical correlation rho (rho2=rho^2)."""
    a=np.sqrt(np.sqrt(rho2))          # corr(g1,x1)=a^2=rho
    si,sc=np.empty(B),np.empty(B)
    for b in range(B):
        Z=rng.standard_normal(nn)
        G=rng.standard_normal((nn,p)); X=rng.standard_normal((nn,d))
        if rho2>0:
            G[:,0]=a*Z+np.sqrt(1-a*a)*rng.standard_normal(nn)
            X[:,0]=a*Z+np.sqrt(1-a*a)*rng.standard_normal(nn)
        G=G*sd_g; X=X*sd_x
        si[b]=stat_in(G,X); sc[b]=stat_cv(G,X)
    return si,sc

IMGS=["tumor_radiomics","organ_radiomics","tumor_radimagenet","whole_radimagenet"]
GRID=[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60]
B=200
rng=np.random.default_rng(0)
sd_g=Gw.std(0)
OUT={}
for IMG in IMGS:
    a=ad.read_h5ad(REPO+f"data/tcga_kirc/imaging/{IMG}.h5ad")[pids]
    Xa=dn(a.X).astype(np.float64);Xa=Xa[:,Xa.std(0)>0];Xa=Xa[:,untied(Xa)]
    Xw=work(Xa,PS); sd_x=Xw.std(0)
    obs_in=stat_in(Gw,Xw); obs_cv=stat_cv(Gw,Xw)
    # --- validation: simulator at rho2=0 vs the REAL permutation null ---
    _,realnull,_=permutation_test(Gw,Xw,n_components=1,n_perm=200,random_state=0)
    realnull=realnull[:,0]**2
    curves={}
    for r2 in GRID:
        si,sc=simulate(n,Gw.shape[1],Xw.shape[1],r2,sd_g,sd_x,B,rng)
        curves[r2]=dict(in_q05=float(np.quantile(si,.05)),in_med=float(np.median(si)),
                        in_q95=float(np.quantile(si,.95)),
                        cv_q05=float(np.quantile(sc,.05)),cv_med=float(np.median(sc)),
                        cv_q95=float(np.quantile(sc,.95)))
    s0=curves[0.0]
    print(f"\n===== {IMG} =====")
    print(f"  simulator rho2=0: in-sample median={s0['in_med']:.3f} q95={s0['in_q95']:.3f}"
          f"   REAL perm null: median={np.median(realnull):.3f} q95={np.quantile(realnull,.95):.3f}")
    print(f"  observed: in-sample={obs_in:.3f}  cv={obs_cv:.3f}")
    for r2 in GRID:
        c=curves[r2]
        print(f"    rho2={r2:.2f}  in q05={c['in_q05']:.3f} med={c['in_med']:.3f} | cv q05={c['cv_q05']:.3f} med={c['cv_med']:.3f}")
    OUT[IMG]=dict(obs_in=obs_in,obs_cv=obs_cv,curves={str(k):v for k,v in curves.items()},
                  real_null_med=float(np.median(realnull)),real_null_q95=float(np.quantile(realnull,.95)))
json.dump(OUT,open("/home/jrich/tmp/claude-1023/-home-jrich-Desktop-rgit/40fecd3b-7b56-47e4-9518-f88caeedf0bc/scratchpad/calib.json","w"),indent=2)
print("\nwrote calib.json")
