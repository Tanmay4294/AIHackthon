# Person 1 Stage 6 Baseline Handoff for Person 2

## Summary of Baseline Findings
- **Recommended Baseline**: `Isolation_Forest` ($F1=0.3851$, Precision=0.3298, Recall=0.4627$).
- **Key Failure Mode of Simple Outlier Rules**: Raw IQR and Z-score methods generate excessive False Positives in Heavy Load regimes ($>85A$).
- **Recommendation for Final Stage 4 Classifier**: Combine Stage 5 engineered features (`p1_max_abs_residual`, `p1_consistency_index`, data-quality flags) into a supervised tree ensemble (e.g. Random Forest / LightGBM) to achieve optimal non-linear separation.
