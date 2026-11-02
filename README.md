# STRIDE Revision — Reproducibility Package

Manuscript: sports-4635610 (Sports, MDPI)  
Round-1 revision re-validation experiments.

This package reproduces the new numerical results added to the manuscript during revision (experiments E1–E10). It does not re-run the original pose-estimation or LSTM training pipelines, because the raw videos and trained model artifacts are not redistributed here.

---

## 1. Contents

```text
Reproducibility_Package/
├── README.md
├── requirements.txt
├── configs/
│   └── experiment_seeds.yaml
├── scripts/
│   ├── exp_part1_verification.py     # E1, E2, E4, E5
│   ├── exp_part2_analyses.py         # E3, E6, E7, E8
│   └── exp_part3_figures.py         # E9, E10
└── outputs/
    └── committed CSV/JSON/PNG results and logs
```

## 2. Source repository

All inputs come from the public repository below:

https://github.com/Shub95-dot/STRIDE-Sensorless-Tracking-for-Risk-and-Injury-Detection-in-Exercise-

The audit used commit `9221feb`.

The scripts assume the repository is checked out as a sibling directory in the following layout:

```text
parent/
├── STRIDE_repo/                    # clone of the original repository
└── STRIDE_Reviewer_Revision/       # this reproducibility package
```

If your clone path differs, update the `REPO` variable at the top of each script.

### Input datasets used

| Experiment | Input file(s) |
|---|---|
| E1 | `15_rehab24_lunge_outputs/biomarkers_per_rep/rehab24_lunge_per_rep_biomarkers.csv` |
| E2 | above + `14_rehab24_outputs/biomarkers_per_rep/rehab24_squat_per_rep_biomarkers.csv` + `phase5b/5c_effect_sizes_ci.csv` |
| E4 | validated LoA constants (Table 2 of the manuscript) + `22_dissertation_writing/appendices/appendix_A_uncertainty_derivation.md` |
| E5 | `20_screening_outputs/worked_example_screening.csv` |
| E6 | per-rep biomarker CSVs for squat and lunge |
| E7 / E8 | per-rep biomarker CSVs + `23_temporal_model_outputs/temporal_model_comparison.csv` |
| E9 / E10 | `16_opencap_dropjump_outputs/metadata/opencap_dropjump_trial_biomarkers.csv` + `phase5b/5c_effect_sizes_ci.csv` |

## 3. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verified on Python 3.12 with:
- numpy 2.1.3
- pandas 2.2.3
- scipy 1.14.1
- scikit-learn 1.5.2

## 4. How to run

Run the scripts in order:

```bash
python3 scripts/exp_part1_verification.py
python3 scripts/exp_part2_analyses.py
python3 scripts/exp_part3_figures.py
```

Expected runtime:
- Part 1: ~3 minutes
- Part 2: ~4 minutes
- Part 3: <1 minute

Each script writes outputs to `STRIDE_Reviewer_Revision/Experimental_Results/` and appends a full run log (`experiment_log_part*.txt`).

## 5. Random seeds

All seeds are fixed in `configs/experiment_seeds.yaml`.

- NumPy / Python seed: 42
- Bootstrap seed: 42
- Per-fold TF seed follows the original repo convention (42 + fold index)

These settings make the stochastic components reproducible. The only random steps are the cluster bootstraps, which are seeded and therefore reproduce the committed CSV outputs exactly.

## 6. Experiment-to-manuscript mapping

| ID | Experiment | Main question addressed | Manuscript location |
|---|---|---|---|
| E1 | Lunge per-subject selection audit + occlusion-bias Fisher test | R1.1, R2.4 | §2.2, new Table 1, §4.1 |
| E2 | Effect-size re-verification (Tables 3–4) | verification of R3.4 | §3.2–3.3 |
| E3 | Unit-sensitivity of inverse-variance weights | R1.2 | §2.4.3, Table 5 footnote |
| E4 | Independent re-run of 9-config variance-split sweep | R1.3 | §2.4.2, Supp. S2 |
| E5 | Counterfactual margin re-verification | R1.4 | §2.7.1, §3.6, §3.6.1 |
| E6 | Baseline-initialization sensitivity + within-subject variability | R1.5, R2.6, R2.7 | §2.5.1, §4.4 |
| E7 | Deterministic LOSO re-run + static baseline comparison | R2.8 | §3.7, Supp. S9 |
| E8 | Subject-level bootstrap confidence intervals for LOSO balanced accuracy | R2.8 | Abstract, §3.7, §5 |
| E9 | Bland–Altman figure with annotated bias and LoA | R3.5 | Figure 2 |
| E10 | Forest plot with 95% CI error bars | R3.6 | Figure 3 |

## 7. What this package intentionally does not re-run

| Item | Reason |
|---|---|
| MediaPipe pose extraction on REHAB24-6 / Penn Action videos | Raw videos are not redistributable; the repo’s verified per-rep biomarker CSVs reproduce the manuscript tables exactly. |
| OpenCap re-processing | Requires the original synced video, IK data, and force data, which are not in the public repo. The values were verified against `phase6_agreement_final.csv` and recomputed from `static_peak_error.csv`. |
| LSTM re-training | TensorFlow is not available in the revision environment; the original per-fold results were verified and the deterministic LOSO baselines were re-run independently. |
| Upstream repository commit | Not done without explicit author authorization. This package is the proposed revision contribution. |

## 8. Provenance and integrity notes

- Every manuscript number checked against the repository outputs matched exactly, except for two prose issues corrected in the revision:
  - §3.6 threshold/baseline labeling was corrected
  - the lunge wording was clarified to distinguish one subject-level exclusion from one trial-level exclusion
- The original supplementary sensitivity table (Table A.1 / S2) could not be reproduced to the last decimal for every configuration; the maximum difference was 6.8 percentage points on the smallest weight. The robustness conclusion was still independently confirmed under two ROM-propagation variants and documented in `outputs/E4_sweep_vs_published.csv`.
- No synthetic or simulated results were used anywhere in the revision.

---

## 9. Summary

This reproducibility package is intended to make the revision audit transparent and checkable. It focuses on the paper’s newly added numbers, figures, and the validation analyses that were added during revision, while avoiding unnecessary re-runs of the full computational pipeline.

If you want to reproduce the revision results, follow the setup and run order above. If you want to inspect a specific experiment, the mapping table in Section 6 links each analysis to its manuscript section and purpose.


