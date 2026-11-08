# Hilbert-space convergence report

Decision: **GO**. CPU QuTiP strict solver, unchanged waveforms and model, N=5,6,7,8.

200 existing selection scenarios, nominal and 10 existing training stress entries were used. Duplicate nominal entries reuse one channel solve. No new uncertainty scenarios or independent validation devices were drawn.

Higher-level relaxation uses n/2 times the n=2 rate and dephasing uses (n/2)^2 times its projector rate, exactly as the existing model. This extrapolation is a model assumption, not measured higher-level noise. The Duffing ladder and annihilation sqrt(n) couplings, six unconditional computational X-state fidelities and total population outside |0>,|1> are preserved.

|Candidate|Max pointwise N5−N8 error|Max N7−N8 error|Five-level check|
|---|---:|---:|---|
|conventional|7.7973719e-09|2.0421387e-10|True|
|risk|4.6975588e-09|2.9469227e-10|True|
|continuation|6.4096439e-09|1.7452084e-10|True|
|calibrated_drag|1.0349607e-08|1.7296542e-10|True|
|risk_unfiltered_diagnostic|4.4977096e-09|1.5784585e-10|True|

All aggregate metrics, signed adjacent differences and signed differences from N8 are in hilbert_convergence.csv. Pointwise tests include saved training stress cases as well as the common 200 selection devices. The existing absolute 2e-6 metric tolerance is used unchanged as a numerical adequacy check. It is not a derived truncation bound or a relative leakage-error guarantee. Report relative errors too when leakage is tiny, and do not infer uniform convergence from this finite sample.

Independent solver verification reports adaptive CPU maximum steps dt/4 versus dt/8 with the same continuous linear waveform interpolant and unchanged atol/rtol. These are not new sampled waveforms. Existing GPU four/eight piecewise interpolation checks remain in the preserved development evidence; QuTiP max-step checks isolate CPU integration error. Selected channels are also checked with six independent QuTiP mesolve state propagations.

An engineering-eligible pulse must separately satisfy nominal fidelity and leakage, sampled ensemble leakage, amplitude/endpoints and the existing post-analysis P05 guard. A high nominal fidelity alone is insufficient. Candidates failing engineering remain diagnostic comparators. If five-level truncation fails, this task does not switch the frozen production model to N8 or silently reoptimize.

Validation protocol, exact waveform arrays, coefficients/native DRAG parameters, costs and source hashes are frozen in this versioned directory. Independent validation has not been executed. Numerical readiness does not establish the RWA/Duffing model, distributions or hardware accuracy, nor Q1 journal readiness.
