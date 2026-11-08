# Initialization study

Mode: **intermediate**. Status: **COMPLETE DEVELOPMENT EXPERIMENT**. This is development on common selection scenarios, not independent validation. No validation dataset was generated.


Qualified starts preserve the amplitude disk, fixed duration and zero endpoints. The unperturbed constrained fit is always start zero; seeded perturbations shrink until nominal fidelity/leakage qualify. Legacy initialization remains available.

| Strategy | Evaluated attempts | Nominally qualified | Minimum F | Maximum F |
|---|---:|---:|---:|---:|
| legacy | 3 | 0 | 0.9670206799 | 0.9902181495 |
| two_harmonic_unperturbed | 3 | 3 | 0.9996804988 | 0.9996804988 |
| five_harmonic_unperturbed | 3 | 3 | 0.9996603009 | 0.9996603009 |
| qualified_fourier_multistart | 15 | 15 | 0.9996272756 | 0.9996760985 |

Qualification frequency is a descriptive initialization result. It does not establish optimization reliability across unseen devices. Rejected attempts and coefficients/hashes are preserved in initialization_results.csv. Its production_channel_solves field for qualified starts is the total for that seed bank, repeated for provenance; it must not be summed across rows.
