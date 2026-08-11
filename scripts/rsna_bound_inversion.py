"""95% upper confidence bound on the POPULATION leading recoverability, by test inversion.

Calibration is SEMI-PARAMETRIC: signal is planted into permuted REAL data, so at rho^2=0
the reference distribution is exactly the permutation null (each modality keeps its own
covariance/tail structure; only the cross-modality link is destroyed).

  G' = G + a*Z a^T,   X' = X_perm + a*Z b^T,   Z ~ N(0,1) independent
  => Sgx' = t*a b^T  (t = alpha^2), and the single population canonical correlation is
     rho = t*sqrt(A*B) / sqrt((1+tA)(1+tB)),  A = a'Sg^-1 a,  B = b'Sx^-1 b
  (Sherman-Morrison).  We bisect t to hit each target rho^2 exactly.

  UB = sup{ rho^2 : q05(stat | rho^2) <= stat_obs }.  Estimator bias is irrelevant.
"""
import sys, json, numpy as np, scipy.sparse as sp
sys.path.insert(0,"/home/jrich/Desktop/rgit-private")
import anndata as ad
from scipy.stats import rankdata
from scipy.special import ndtri
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rgit import fit_recoverability
REPO="/home/jrich/Desktop/rgit-private/"
SD="/home/jrich/tmp/claude-1023/-home-jrich-Desktop-rgit/40fecd3b-7b56-47e4-9518-f88caeedf0bc/scratchpad/"
dn=lambda M: M.toarray() if sp.issparse(M) else np.asarray(M)
def grt(M):
    M=np.asarray(M,float);n=M.shape[0]
    return ndtri(np.apply_along_axis(lambda c: rankdata(c,method="average"),0,M)/(n+1.0))
def untied(M,mt=0.5):
    return np.array([np.unique(M[:,j],return_counts=True)[1].max()/M.shape[0]<=mt for j in range(M.shape[1])])
stat=lambda G,X: float(fit_recoverability(G,X,n_components=1).recoverability[0])

def rho_of_t(t,A,B):
    return t*np.sqrt(A*B)/np.sqrt((1+t*A)*(1+t*B))
def solve_t(target_rho,A,B):
    lo,hi=0.0,1.0
    while rho_of_t(hi,A,B)<target_rho and hi<1e12: hi*=2
    for _ in range(200):
        mid=(lo+hi)/2
        if rho_of_t(mid,A,B)<target_rho: lo=mid
        else: hi=mid
    return (lo+hi)/2

g=ad.read_h5ad(REPO+"data/tcga_kirc/genomics/gene_expression.h5ad")
Gl=dn(g.layers["tpm_unstranded"]).astype(np.float64)
libs=Gl.sum(1,keepdims=True);libs[libs==0]=1.0
Gl=np.log1p(Gl/libs*np.median(libs))
ok=np.isfinite(Gl).all(0)&((Gl>0).mean(0)>0.1);Gl=Gl[:,ok];Gl=Gl[:,untied(Gl)]
pids=list(g.obs_names);n=len(pids)
hv=np.argsort(Gl.var(0))[::-1][:2000]
Gz=StandardScaler().fit_transform(grt(Gl[:,hv]))
pg=PCA(min(60,n-1),random_state=0).fit(Gz); Gall=pg.transform(Gz)
evr=np.cumsum(pg.explained_variance_ratio_)

GRID=np.array([0.0,0.02,0.04,0.06,0.08,0.10,0.13,0.16,0.20,0.25,0.30,0.38,0.46,0.55])
B, NDIR = 240, 12
OUT={}
for IMG in ["tumor_radiomics","organ_radiomics","tumor_radimagenet","whole_radimagenet"]:
    a_=ad.read_h5ad(REPO+f"data/tcga_kirc/imaging/{IMG}.h5ad")[pids]
    Xa=dn(a_.X).astype(np.float64);Xa=Xa[:,Xa.std(0)>0];Xa=Xa[:,untied(Xa)]
    Xz=StandardScaler().fit_transform(grt(Xa))
    Xall=PCA(min(60,n-1,Xz.shape[1]),random_state=0).fit_transform(Xz)
    OUT[IMG]={}
    for PS in [5,10,20,38]:
        G,X=Gall[:,:PS],Xall[:,:PS]
        obs=stat(G,X)
        Sg,Sx=np.cov(G.T),np.cov(X.T)
        Sgi,Sxi=np.linalg.inv(Sg),np.linalg.inv(Sx)
        rng=np.random.default_rng(0)
        q05=[];med=[]
        for r2 in GRID:
            target=np.sqrt(r2); s=[]
            for d_ in range(NDIR):
                av=rng.standard_normal(PS); av/=np.linalg.norm(av)
                bv=rng.standard_normal(X.shape[1]); bv/=np.linalg.norm(bv)
                A=float(av@Sgi@av); Bq=float(bv@Sxi@bv)
                t=0.0 if r2==0 else solve_t(target,A,Bq)
                al=np.sqrt(t)
                for _ in range(B//NDIR):
                    Xp=X[rng.permutation(n)]
                    Z=rng.standard_normal(n)
                    s.append(stat(G+al*np.outer(Z,av), Xp+al*np.outer(Z,bv)))
            s=np.array(s); q05.append(np.quantile(s,.05)); med.append(np.median(s))
        q05=np.array(q05)
        ub=0.0 if obs<q05[0] else (float(GRID[-1]) if obs>=q05[-1] else float(np.interp(obs,q05,GRID)))
        OUT[IMG][PS]=dict(obs=obs,ub95=ub,null_med=float(med[0]),null_q05=float(q05[0]),
                          var_captured=float(evr[PS-1]),q05=q05.tolist(),med=med)
        print(f"[{IMG:20s} p*={PS:2d}] obs={obs:.3f} null_med={med[0]:.3f} "
              f"UB95(rho1^2)={ub:.3f}  genomic var captured={evr[PS-1]*100:.0f}%")
    print("")
json.dump({"grid":GRID.tolist(),"res":{k:{str(a):b for a,b in v.items()} for k,v in OUT.items()}},
          open(SD+"invert2.json","w"),indent=2)
print("wrote invert2.json")
