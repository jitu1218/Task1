# Validation

Training and scoring executed successfully in the build environment.
Nine unit/integration tests passed, covering ECE calculations and boundaries,
invalid inputs, disjoint splits, validation-based selection, reproduced test metrics,
and saved-model scoring consistency. The two PNG figures were generated; the
reliability plot was visually inspected.

Fixed-seed results:
- Selected by validation Brier: random forest with sigmoid calibration.
- Test ECE: 0.0781 raw -> 0.0207 sigmoid.
- Test Brier: 0.0742 raw -> 0.0653 sigmoid.
- Test ROC-AUC: 0.9562 for both.

Primary environment packages: numpy 2.3.5, scikit-learn 1.8.0,
matplotlib 3.10.8, joblib 1.5.3. Windows commands are supplied; execution was tested
on Linux, not directly on Windows. No real-world deployment performance is claimed.
