# Confirmatory execution implementation audit — version 1

**BLOCKED BEFORE VALIDATION GENERATION.** The scientific freeze is intact, but
its manifest contains no complete confirmatory execution entry point. The prior
freeze included `confirmatory_statistics.py`, a primary bootstrap helper, and
`preconfirmatory_checks.py`, a development-only200-device checking program.
Neither is an executable2000-device confirmatory campaign. `study_validation.py`
is a general diagnostics library and is not a substitute for the frozen experiment.
The prior step froze the scientific plan but omitted a complete executor.

The user's Phase1 instruction explicitly says: “If a frozen confirmatory
implementation is missing, do not silently substitute the development benchmark.”
It then requires reporting the blocker and preparing a separate versioned
implementation that can be audited before any validation devices are generated.
This package implements that branch. No original file has been repaired/replaced.

## Scientific freeze and filename resolution

4383 frozen artifacts were checksum-verified. Every required root source and the
archived audit dependencies are verified against the frozen snapshots; package
versions must match. All five waveform hashes, roles, fixed physical parameters,
uniform uncertainty ranges, seeds, sample counts, strict solver settings and the
six contrast definitions match the scientific protocol. Historical
`validation_execution_authorized=false` records describe the previous preparation
task; they are not interpreted as revoking the user's new execution request.
The blocker is the missing originally frozen executor and its pre-data audit.

The requested protocol/configuration/analysis-plan/solver report filenames are
resolved to the existing immutable `confirmatory_protocol_frozen.md`,
`confirmatory_protocol.json`, its statistical-analysis section, and
`independent_solver_verification.json`. The mapping is saved separately rather
than renaming frozen files.

Older unrelated validation datasets exist from earlier smoke/research campaigns.
Their recorded seeds belong to those older studies. They will not be reused.
No output cohort using reserved seed941206 has been identified; no reserved
cohort was generated or inspected during this audit. Absence of exact duplicate
scenario rows is a future consistency check, not proof of statistical independence.

## Proposed implementation

- `confirmatory_execution_v1.py` defaults to audit-only. `--run` additionally
  requires a review record accepting the exact implementation manifest and
  scientific freeze. No automatic acceptance option exists.
- Five immutable waveforms use the already frozen CPU production kernel, without
  fitting, scaling or optimization. The diagnostic remains outside primary
  engineering eligibility.
- Exactly2000 devices, seed941206, and the frozen generator/distributions are
  fixed constants. A single global cohort reservation prevents changing output
  directories from creating another cohort. Dataset writes are atomic, hashed
  and read-only. An interrupted pre-commit generation fails closed, not redraws.
- Each candidate-device checkpoint records cohort/pulse/scenario identity, six
  state fidelities/leakages, physicality diagnostics and a payload checksum.
  Completed results are verified before reuse. Solver failures stop confirmation;
  returned in-flight results and failures are preserved, without favorable filtering.
- All10000 production evaluations precede the frozen20-index finer-step and
  state-propagation checks. The primary frozen bootstrap helper is reused exactly
  for50000 and100000 replicates, seed941207, six contrasts and Bonferroni CIs.
  Secondary percentile intervals replay the same paired index stream.
- Existing one-sided95% P05 certification is reused unchanged; the simultaneous
  four-primary-pulse certificate uses the same method at1−.05/4 confidence.
- Reports include all negative/mixed outcomes. Seven figure types and the requested
  CSV/JSON outputs are implemented. A complete numerical simulation does not imply
  experimental hardware evidence or Q1-journal acceptance.

## Items requiring explicit conformance review before release

The frozen text provides numerical-resolution cautions and the three possible
publication-readiness labels, but no unique mechanical global-label algorithm.
The proposed adapter exposes, rather than hides, these operational interpretations:

1. CIs near the existing2e-6 resolution scale receive a numerical-review flag;
   the primary CI and its statistical direction are never changed. Observed
   repeat-device numerical envelopes are finite sensitivity evidence, not a
   uniform CVaR error bound.
2. Its proposed overall label counts favorable, sensitivity-stable, engineering-
   passing contrasts across the six directed comparisons. Every individual
   contrast remains authoritative. This proposed ordinal summary must be reviewed
   before any data, rather than adopted after outcomes. It does not imply that a
   negative research result is scientifically unsuccessful.

These are proposed implementation interpretations, not claimed original frozen
rules. Review must either accept fidelity to the existing plan or explicitly
resolve its interpretation before data; changes require a new implementation
manifest. Never decide them after independent outcomes exist.

## Tests and execution status

17 relevant automated tests passed: source/release hash binding, no generation
without review, no redraw after interruption/corruption, verified cohort reuse,
checkpoint corruption/failure rejection, exact100-device CVaR tail at n2000,
paired resampling, unchanged dimension/dissipator checks and diagnostic exclusion.
Tests use schema fixtures and synthetic mathematical vectors in temporary
folders, not independent physical validation devices or fabricated study results.

The real audit also reproduces all five nominal channels and compares CPU
four/eight maximum-step and six-state propagation on nominal and the two saved
joint stress cases. These use development cases only, not a pilot validation set.

**Independent devices generated:0/2000. Candidate-device validation evaluations:
0/10000. No confirmatory scientific verdict has been issued.**

## Exact execution command after implementation review and release

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/kairos-mpl .venv/bin/python -u confirmatory_execution_v1.py --run --output results/confirmatory_validation_v1
```

Use the same command after interruption; it verifies and reuses the single saved
cohort and completed checkpoints. Missing/corrupt data, failures or version/hash
mismatches block execution. Review data bind to `implementation_manifest.json`;
`implementation_review.template.json` is not an acceptance record and is never
read as one by the executor. No validation run was launched in this preparation.
