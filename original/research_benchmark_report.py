"""Research artifacts with separation of tuning data, optimizer spread and device uncertainty."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(out,result,sensitivity,costs,values,maps):
    out=Path(out);figdir=out/'publication_figures';figdir.mkdir(exist_ok=True)
    def save(fig,i):fig.tight_layout();fig.savefig(figdir/f'figure_{i}.png',dpi=180,bbox_inches='tight');plt.close(fig)
    for i,metric,label in [(1,'p05','Selection empirical P05 fidelity'),(2,'cvar95','Selection empirical CVaR95 infidelity')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for regime in result['effective_settings']['regimes']:
            rows=[r for r in sensitivity if r['regime']==regime and r['phase']=='selection_only']
            ax.plot([r['lambda_value'] for r in rows],[r[metric] for r in rows],'.-',label=regime)
        ax.set(xlabel='CVaR weight lambda (dimensionless)',ylabel=label,title='Selection data only; not held-out performance');
        if metric=='p05':ax.set_ylim(0,1)
        ax.legend();save(fig,i)
    fig,ax=plt.subplots(figsize=(8,5))
    for key,arrays in values.items():
        if key.endswith(('_risk','_conventional','_drag')):
            f=np.sort(np.asarray(arrays).ravel());ax.step(f,np.arange(1,len(f)+1)/len(f),where='post',label=key)
    ax.set(xlabel='Held-out unconditional gate fidelity',ylabel='Empirical probability',title='Pooled display only; inference retains trial/device structure');ax.legend(fontsize=6);save(fig,3)
    fig,ax=plt.subplots(figsize=(8,5))
    for regime in result['comparisons']:
        d=(np.array(values[regime+'_risk'])-np.array(values[regime+'_conventional'])).mean(axis=0)
        ax.hist(d,bins=15,alpha=.5,label=regime)
    ax.axvline(0,color='k',ls=':');ax.set(xlabel='Paired fidelity difference averaged over trials (risk - conventional)',ylabel='Device count');ax.legend();save(fig,4)
    fig,axes=plt.subplots(1,max(1,len(maps)),figsize=(max(8,5*len(maps)),5),squeeze=False)
    for ax,(key,rows) in zip(axes.flat,maps.items()):
        n=int(np.sqrt(len(rows)));x=np.array([r['detuning_error_mhz'] for r in rows]).reshape(n,n);y=np.array([r['amplitude_scale'] for r in rows]).reshape(n,n)
        z=np.array([1-r['fidelity'] for r in rows]).reshape(n,n)
        mesh=ax.pcolormesh(x,y,z,shading='auto',vmin=0,vmax=max(1e-2,max(1-r['fidelity'] for rows2 in maps.values() for r in rows2)))
        fig.colorbar(mesh,ax=ax,label='Unconditional gate infidelity (1-F)')
        leak=np.array([r['leakage']>1e-4 for r in rows]).reshape(n,n)
        if leak.any() and not leak.all():ax.contour(x,y,leak,levels=[.5],colors='red')
        ax.set(title=key.replace('equal_evaluations','Equal calls\n').replace('equal_wall','Equal wall cap\n').replace('_',' '),xlabel='Detuning error (MHz)',ylabel='Amplitude multiplier')
        ax.title.set_fontsize(10)
        ax.set_xlim(float(x.min()),float(x.max()));ax.set_ylim(float(y.min()),float(y.max()))
    save(fig,5)
    fig,ax=plt.subplots(figsize=(8,5))
    for key,rows in result['results'].items():
        if '_reference_' not in key:
            ax.scatter([r['statistics']['worst_sample_leakage'] for r in rows],[r['statistics']['mean_fidelity'] for r in rows],label=key)
    ax.set(xlabel='Maximum observed leakage probability',ylabel='Mean held-out fidelity',title='Diagnostic trade-off scatter; not a Pareto frontier');ax.legend(fontsize=5,ncol=2);save(fig,6)
    fig,ax=plt.subplots(figsize=(9,5))
    for file in out.glob('equal*_*.json'):
        obj=json.loads(file.read_text())
        if isinstance(obj,dict) and obj.get('name') in ('conventional',) or isinstance(obj,dict) and str(obj.get('name','')).startswith('mu_'):
            h=obj.get('history',[])
            if h:ax.plot([r.get('iteration',0) for r in h],[r['loss'] for r in h],label=file.stem,alpha=.6)
    ax.set(xlabel='Recorded optimizer step',ylabel='Own objective (objectives differ)',title='No direct comparison of unlike objective magnitudes');ax.legend(fontsize=5,ncol=2);save(fig,7)
    fig,ax=plt.subplots(figsize=(8,5))
    for key,rows in result['results'].items():
        if key.endswith(('_risk','_conventional','_nm')):
            regime=key.rsplit('_',1)[0];method=key.rsplit('_',1)[1]
            cs=[r for r in costs if r['regime']==regime and (r['method']=='conventional' if method=='conventional' else r['method']=='nm' if method=='nm' else r['method'].startswith('mu_'))]
            ax.scatter([sum(r['elapsed_seconds'] for r in cs)]*len(rows),[r['statistics']['cvar95_infidelity'] for r in rows],label=key)
    ax.set(xlabel='Measured campaign optimization/selection time (s)',ylabel='Held-out CVaR95 infidelity',title='Actual costs include tuning trials for risk method');ax.legend(fontsize=6);save(fig,8)
    lines=['# Kairos research benchmark','','## Research question','','Does risk-aware GRAPE reproducibly reduce CVaR95 infidelity relative to conventional GRAPE?',
        '','## Mathematical methodology','','Mean, smoothed CVaR and leakage objectives use the existing dissipative adjoint. Worst-loss ablation uses log-sum-exp with error <= tau*log(N). CVaR softplus error <= tau*log(2)/(1-alpha).',
        '','## Physical model','','Unchanged five-level Duffing/RWA, Lindblad noise, six-state unconditional fidelity, and waveform constraints. These are simulations, not hardware measurements.',
        '','## Experimental configuration','',f"Mode: **{result['mode']}**. Effective settings: `{result['effective_settings']}`.",
        '','## Optimization budgets','','Separate equal-objective-call and equal-wall-cap regimes. Wall caps are checked between expensive evaluations and may overshoot by one evaluation. Actual costs and early stopping are reported, not assumed equal. Baseline calibration is a distinct one-off expense. Risk tuning uses more total runs and its aggregate cost is explicit in optimization_costs.csv.',
        '','## Statistical methodology','','Hypotheses and comparisons were saved in protocol.json before validation. Lambda then mu use selection data only. All waveforms were hashed before validation draws were generated. Paired crossed bootstrap resamples common trial indices and common device indices separately. Mean per-trial CVaR difference is primary (negative favors risk); P05 is exploratory secondary. Bonferroni adjustment covers the two primary budget-regime comparisons. Percentile bootstrap intervals are approximate, not exact global guarantees. Absolute P05 certification retains exact binomial bounds.',
        '','## Results','','| Regime | CVaR difference risk-conventional | Paired bootstrap interval | Superiority supported |','|---|---:|---|---|']
    for key,row in result['comparisons'].items():lines.append(f"| {key} | {row['cvar_difference']:.9g} | {row['cvar_ci']} | {row['superiority_supported']} |")
    lines+=['','| Frozen group | Trials | Mean empirical CVaR | Engineering passes |','|---|---:|---:|---:|']
    for key,rows in result['results'].items():lines.append(f"| {key} | {len(rows)} | {np.mean([r['statistics']['cvar95_infidelity'] for r in rows]):.9g} | {sum(r['engineering']['passed'] for r in rows)} |")
    lines+=['','## Ablation analysis','','A: mean only; B: smoothed worst infidelity; C: CVaR only; D: mean+CVaR; E: mean+CVaR+leakage. Hardware/qualification constraints remain active even when objective leakage weight is zero. Compare saved groups; a short single-seed pilot cannot establish component contributions.',
        '','## Limitations','',result['remaining'],'Finite numerical robustness searches are not global bounds. Tail bootstrap reliability needs adequate device/trial counts. Initialization spread and Monte Carlo spread must not be conflated. No quantum advantage or CVaR novelty is asserted.',
        '','## Reproducibility instructions','','`python research_benchmark.py --config research_config.json --output results/research_full`',
        '`python research_benchmark.py --config research_config.json --quick --output results/research_quick`',
        'Re-run the same command to resume. Source/config hashes must match; changed protocols need a new directory. --max-new-runs checkpoints an incomplete training campaign without generating validation data. --prepare-only saves the full protocol without running it.',
        '','## Publication figures','',*[f'![Figure {i}](publication_figures/figure_{i}.png)' for i in range(1,9)]]
    (out/'research_benchmark_summary.md').write_text('\n'.join(lines)+'\n')
