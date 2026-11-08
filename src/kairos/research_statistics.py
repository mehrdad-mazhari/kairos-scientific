"""Paired crossed bootstrap: optimizer-trial and device variability remain distinct."""
import numpy as np
from .risk_aware_grape import empirical_cvar


def paired_bootstrap(risk, conventional, *, repetitions=2000, seed=6300, family_size=2):
    a=np.asarray(risk,dtype=float);b=np.asarray(conventional,dtype=float)
    if a.shape!=b.shape or a.ndim!=2 or min(a.shape)<1 or a.shape[1]<2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('paired finite trial-by-device fidelity matrices required')
    if repetitions<100 or family_size<1:raise ValueError('insufficient bootstrap repetitions or invalid family')
    def metrics(x,y):
        return np.array([np.mean([empirical_cvar(1-u)-empirical_cvar(1-v) for u,v in zip(x,y)]),
                         np.mean(np.quantile(x,.05,axis=1)-np.quantile(y,.05,axis=1)),(x-y).mean()])
    point=metrics(a,b);rng=np.random.default_rng(seed);samples=[]
    for _ in range(repetitions):
        trials=rng.integers(a.shape[0],size=a.shape[0]);devices=rng.integers(a.shape[1],size=a.shape[1])
        samples.append(metrics(a[trials][:,devices],b[trials][:,devices]))
    samples=np.array(samples);level=.05/family_size
    intervals=np.quantile(samples,[level/2,1-level/2],axis=0)
    supported=a.shape[0]>=10 and a.shape[1]>=2000
    return dict(estimand='Mean over paired optimizer trials of per-trial device CVaR95 difference (risk minus conventional)',
        cvar_difference=float(point[0]),cvar_ci=intervals[:,0].tolist(),p05_difference=float(point[1]),p05_ci=intervals[:,1].tolist(),
        mean_difference=float(point[2]),mean_ci=intervals[:,2].tolist(),optimizer_trials=a.shape[0],devices=a.shape[1],
        bootstrap_repetitions=repetitions,bootstrap_seed=seed,family_size=family_size,
        method='Paired crossed percentile bootstrap over trials and common device indices; Bonferroni family adjustment; asymptotic, not exact',
        primary_hypothesis='H0: CVaR(risk)-CVaR(conventional)>=0; H1: difference<0',
        significance_level=.05,exploratory=not supported,
        superiority_supported=bool(supported and intervals[1,0]<0),
        interpretation='No automatic claim from marginal CI overlap. Bootstrap tail coverage is approximate; smoke intervals are exploratory.')
