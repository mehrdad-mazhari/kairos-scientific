"""Dissipative Fourier GRAPE with infidelity CVaR (independent of certification)."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logsumexp
from .grape import EnsembleGRAPE, settings as grape_settings, refine as conventional_refine

DEFAULTS=dict(enabled=False,alpha=.95,risk_weight=1.,leakage_weight=2.,smoothing=1e-6,
              training_samples=32,training_seed=2050,optimizer_seeds=[2030,2031])


def settings(config):
    return {**grape_settings(config),**DEFAULTS,**config.get('risk_aware_grape',{})}


def empirical_cvar(losses,alpha=.95):
    values=np.sort(np.asarray(losses,dtype=float))[::-1]
    if values.ndim!=1 or not len(values) or not np.all(np.isfinite(values)) or not 0<=alpha<1:
        raise ValueError('require finite nonempty losses and 0<=alpha<1')
    mass=(1-alpha)*len(values)
    weights=np.clip(mass-np.arange(len(values)),0.,1.)/mass
    return float(values@weights)


def smooth_cvar(losses,alpha=.95,tau=1e-6):
    """Minimize the softplus Rockafellar–Uryasev envelope; return dCVaR/dloss.

    Exact CVaR <= smoothed CVaR <= exact CVaR + tau*log(2)/(1-alpha).
    The eta envelope derivative vanishes at the scalar root, so the analytic
    loss gradient is sigmoid((loss-eta)/tau)/((1-alpha)*N).
    """
    x=np.asarray(losses,dtype=float)
    empirical_cvar(x,alpha)
    if not np.isfinite(tau) or tau<=0:raise ValueError('smoothing must be finite and positive')
    if alpha==0:return float(x.mean()),np.ones(len(x))/len(x),float(x.min())
    # Center/scale for stable roots even when tau is very small.
    center=float(x.max());z=(x-center)/tau
    margin=max(40.,-np.log1p(-alpha)+10.)
    root=brentq(lambda eta:expit(z-eta).mean()-(1-alpha),float(z.min()-margin),margin,xtol=1e-12)
    eta=center+tau*root
    value=eta+tau*np.logaddexp(0.,z-root).mean()/(1-alpha)
    gradient=expit(z-root)/((1-alpha)*len(x))
    return float(value),gradient,float(eta)


class RiskEnsembleGRAPE(EnsembleGRAPE):
    def value_gradient(self,controls):
        losses=[];leaks=[];gf=[];gl=[]
        # Reuse cached invariant Liouvillians and established dissipative adjoint.
        # Two passes separate fidelity and leakage; no closed-system surrogate.
        saved=self.options['leakage_weight']
        try:
            for model in self.models:
                self.options['leakage_weight']=0.
                f,df=self.scenario_value_gradient(controls,model)
                self.options['leakage_weight']=1.
                combined,dcombined=self.scenario_value_gradient(controls,model)
                losses.append(f);leaks.append(combined-f);gf.append(df);gl.append(dcombined-df)
        finally:self.options['leakage_weight']=saved
        losses=np.array(losses);gf=np.array(gf);gl=np.array(gl)
        cv,weights,_=smooth_cvar(losses,self.options['alpha'],self.options['smoothing'])
        lam=self.options['risk_weight'];mu=saved
        mean_weight=self.options.get('mean_weight',1.)
        worst_weight=self.options.get('worst_weight',0.)
        tau=self.options['smoothing']
        worst_weights=np.exp((losses-losses.max())/tau-logsumexp((losses-losses.max())/tau))
        smooth_worst=float(losses.max()+tau*logsumexp((losses-losses.max())/tau))
        value=mean_weight*losses.mean()+lam*cv+mu*np.mean(leaks)+worst_weight*smooth_worst
        gradient=mean_weight*gf.mean(axis=0)+lam*np.einsum('i,ijk->jk',weights,gf)+mu*gl.mean(axis=0)+worst_weight*np.einsum('i,ijk->jk',worst_weights,gf)
        differences=np.diff(controls,axis=0);weight=self.options['smoothness_weight']/len(differences)
        value+=weight*np.sum(differences**2)
        gradient[:-1]-=2*weight*differences;gradient[1:]+=2*weight*differences
        self.last_risk=dict(mean_infidelity=float(losses.mean()),exact_cvar=empirical_cvar(losses,self.options['alpha']),
            smoothed_cvar=cv,smoothing_error_bound=self.options['smoothing']*np.log(2)/(1-self.options['alpha']))
        self.last_ensemble_metrics=dict(mean_training_fidelity=float(1-losses.mean()),
            cvar95_training_infidelity=empirical_cvar(losses),smooth_cvar95_training_infidelity=cv,
            mean_training_leakage=float(np.mean(leaks)),maximum_training_leakage=float(np.max(leaks)))
        return float(value),gradient


def refine(pulse,device,scenarios,options,amplitude_scale):
    return conventional_refine(pulse,device,scenarios,dict(options),amplitude_scale,model_class=RiskEnsembleGRAPE)
