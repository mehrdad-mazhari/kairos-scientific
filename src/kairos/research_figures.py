"""Paired exploratory comparisons and seven publication-oriented diagnostic figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from .risk_aware_grape import empirical_cvar


def save_research_outputs(output,manifest,mc_rows,public_rows):
    output=Path(output)
    keys=[k for k in ('drag_baseline_calibrated','nelder_mead','grape_proposal','risk_aware_grape') if k in mc_rows]
    reference=keys[0] if keys else 'robust_selected'
    base=mc_rows[reference];comparisons={}
    for key in keys:
        rows=mc_rows[key];d=np.array([r['fidelity']-b['fidelity'] for r,b in zip(rows,base)])
        f=np.array([r['fidelity'] for r in rows]);bf=np.array([r['fidelity'] for r in base])
        # Bonferroni simultaneous t intervals for a predeclared finite mean family,
        # still approximate and exploratory (no tail-distribution interval claimed).
        half=float(t.ppf(1-.05/(2*max(1,len(keys))),len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)))
        comparisons[key]=dict(reference=reference,mean_fidelity_difference=float(d.mean()),
            approximate_bonferroni_mean_ci95=[float(d.mean()-half),float(d.mean()+half)],
            empirical_p05_difference=float(np.quantile(f,.05)-np.quantile(bf,.05)),
            cvar95_infidelity_difference=empirical_cvar(1-f)-empirical_cvar(1-bf),
            mean_leakage_difference=float(np.mean([r['leakage']-b['leakage'] for r,b in zip(rows,base)])),
            superiority_claim=False,scope='Exploratory; tail differences have no confidence interval. No independent-restart validation study completed.')
    manifest['research_comparisons']=comparisons
    (output/'research_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    def finish(fig,i):fig.savefig(output/f'research_figure_{i}.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,3));ax.axis('off')
    ax.text(.5,.5,'Fixed training → NM / conventional GRAPE / risk-aware GRAPE\n↓\nCommon selection draws → FREEZE + SHA256\n↓\nIndependent validation → exact P05 certification → fixed-pulse diagnostics',ha='center',va='center',fontsize=12)
    finish(fig,1)
    fig,ax=plt.subplots(figsize=(9,5))
    for name in ('grape','risk_aware_grape'):
        for run in manifest.get(name,{}).get('runs',[]):
            h=run['history'];ax.plot([r['iteration'] for r in h],[r['loss'] for r in h],label=f'{name}, seed {run["seed"]}')
    ax.set(xlabel='Recorded optimizer step',ylabel='Own training objective (different objectives)',title='Convergence; objective values are not directly comparable');ax.legend(fontsize=7);finish(fig,2)
    fig,ax=plt.subplots(figsize=(9,5))
    for key in keys:
        f=np.sort([r['fidelity'] for r in mc_rows[key]]);ax.step(f,np.arange(1,len(f)+1)/len(f),label=key,where='post')
    ax.set(xlabel='Held-out unconditional X fidelity',ylabel='Empirical cumulative probability');ax.legend(fontsize=7);finish(fig,3)
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,metric,title in zip(axes,('p05_fidelity','cvar95_infidelity'),('Empirical P05 fidelity','Empirical CVaR95 infidelity')):
        ax.bar(range(len(keys)),[manifest['variants'][k]['monte_carlo'][metric] for k in keys]);ax.set_xticks(range(len(keys)),keys,rotation=40,ha='right',fontsize=7);ax.set_title(title)
    fig.tight_layout();finish(fig,4)
    fig,ax=plt.subplots(figsize=(8,5))
    for k in keys:ax.scatter([r['leakage'] for r in mc_rows[k]],[r['fidelity'] for r in mc_rows[k]],label=k,s=12)
    ax.set(xlabel='Leakage probability (n ≥ 2)',ylabel='Unconditional X fidelity',title='Device diagnostic scatter — not a Pareto frontier');ax.legend(fontsize=7);finish(fig,5)
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,axis in zip(axes,('amplitude_scale','detuning_mhz')):
        for k in keys:
            rows=[r for r in public_rows[k] if r['alpha_shift_mhz']==0 and (r['detuning_mhz']==0 if axis=='amplitude_scale' else r['amplitude_scale']==1)]
            rows=sorted(rows,key=lambda r:r[axis]);ax.plot([r[axis] for r in rows],[r['fidelity'] for r in rows],'.-',label=k)
        ax.set(xlabel='Amplitude scale (dimensionless)' if axis=='amplitude_scale' else 'Additive detuning error (MHz)',ylabel='Fidelity');ax.legend(fontsize=6)
    finish(fig,6)
    fig,ax=plt.subplots(figsize=(9,5))
    for k,method in [('grape_proposal','grape'),('risk_aware_grape','risk_aware_grape')]:
        if k in keys:
            cost=sum(r['history'][-1].get('elapsed_seconds',0.) for r in manifest.get(method,{}).get('runs',[]) if r['history'])
            ax.scatter(cost,manifest['variants'][k]['monte_carlo']['cvar95_infidelity'],label=k)
    ax.set(xlabel='Total optimization wall time over seeds (s)',ylabel='Held-out empirical CVaR95 infidelity',title='Representative training-selected proposals; no superiority claim');ax.legend(fontsize=7);finish(fig,7)
