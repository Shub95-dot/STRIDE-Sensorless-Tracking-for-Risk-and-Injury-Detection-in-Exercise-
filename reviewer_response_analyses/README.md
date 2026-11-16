# Reviewer response analyses

These are robustness analyses run in response to peer-review comments (E1 noise-floor recomputation, E2 downsampling robustness, E3 threshold sensitivity, E4 supervised vs label-free comparison, E5 bootstrap/per-participant/out-of-fold checks, and E7 lunge LOSO effect sizes).

Files in this folder:
- reviewer_experiments.py — Reviewer-response experiments for the second-round review.
- experiment_log.txt — E1. VERIFICATION OF THE UNCERTAINTY CHAIN and related reviewer-response analyses.
- results_summary.json — JSON summary of the reviewer-response experiment outputs keyed by E1 through E7.
- E2_downsample_lunge.csv — Downsampling sensitivity results for lunge trajectories at 30 Hz versus 15 Hz.
- E2_downsample_squat.csv — Downsampling sensitivity results for squat trajectories at 30 Hz versus 15 Hz.
- E3_threshold_sensitivity.csv — Threshold sensitivity results for the uncertainty-propagation checks.
- E4_supervised_vs_labelfree.csv — Comparison of supervised and label-free rules.
- E5_bootstrap_variants.csv — Bootstrap confidence intervals for the peak-flexion baseline.
- E5_oof_predictions_lunge.csv — Out-of-fold prediction table for lunge samples.
- E5_oof_predictions_squat.csv — Out-of-fold prediction table for squat samples.
- E5_per_participant_lunge.csv — Per-participant confusion counts and balanced accuracy for lunge analyses.
- E5_per_participant_squat.csv — Per-participant confusion counts and balanced accuracy for squat analyses.
- E7_lunge_LOSO_effect_sizes.csv — Leave-one-subject-out effect sizes for the lunge analysis.

To rerun the analysis: `python reviewer_experiments.py`.

Before running, check the script imports and the repository/data paths at the top of the file, and note any dependencies or input locations that it expects. Do not guess the file layout or data locations. The generated summary is in `results_summary.json`.
