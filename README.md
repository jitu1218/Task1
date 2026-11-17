# Build a Calibrated Probability Scoring Model

Complete assignment with Python source, fitted scorer, comparison plots, calculated
metrics, a brief report, and reproducible tests. Dataset: 10,000 synthetic binary
classification examples generated locally; no dataset download is needed.

## Windows quick start

Use Python **3.11 or 3.12**. Extract the ZIP and open PowerShell inside
`calibrated_scoring`, where `train.py` is located.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python train.py
.venv\Scripts\python score.py
.venv\Scripts\python -m unittest discover -s tests -v
```

On macOS/Linux replace `.venv\Scripts\python` with `.venv/bin/python`.
Using the environment's Python directly avoids PowerShell activation restrictions.
The first installation needs internet access; subsequent training runs work offline.
Training normally takes under a few minutes, depending on your computer.

Already generated results are included. Open **`outputs/REPORT.html`** in a browser
to read the report and view both figures without running Python. `REPORT.md` is the
editable Markdown version. HTML embeds the plots and can be shared as a single file.

## What it demonstrates

- Logistic regression and a random forest, each producing P(class 1).
- Raw, sigmoid (Platt-style), and isotonic versions: six comparisons.
- Training/calibration/validation/test = 50%/20%/15%/15%, all stratified and disjoint.
- Base estimators and scaler fitted only on training data.
- FrozenEstimator ensures calibration doesn't retrain the base classifier.
- Model selection uses validation Brier score, not final test metrics.
- Final evaluation: ECE at 5/10/20 bins, Brier, log loss and ROC-AUC.
- Reliability curves and probability histograms, plus metric comparison charts.

The selected scorer for the supplied run is **Random forest / sigmoid**. Isotonic
has slightly better test ECE/Brier in this run, but switching after examining the
test set would contaminate the selection process. The saved selection is unchanged.

## Output files

| File | Purpose |
|---|---|
| `outputs/REPORT.html` | Standalone illustrated report |
| `outputs/REPORT.md` | Brief editable report |
| `outputs/reliability_diagrams.png` | Raw vs calibrated reliability and probability distributions |
| `outputs/metric_comparison.png` | ECE, Brier and log-loss comparison |
| `outputs/metrics.csv` | All six candidates evaluated on test data |
| `outputs/validation_metrics.csv` | Evidence supporting model selection |
| `outputs/test_predictions.csv` | Test labels and all six probability estimates |
| `outputs/reliability_bins.csv` | Per-bin means, outcome rates, and support |
| `outputs/manifest.json` | Seed, chosen model and split sample indices |
| `outputs/scorer.joblib` | Fitted validation-selected model |
| `outputs/example_features.csv` | Five example rows for scoring |
| `outputs/new_scores.csv` | Scores produced by score.py |

`train.py --seed 42 --output outputs` recreates the results. Other seeds can produce
different findings. The source fixes hyperparameters; test data is never used to tune
or fit a model. `requirements.txt` pins the primary packages used for the build.
Dependency subversions/hardware may still cause small numerical differences.

## Score new rows

```powershell
.venv\Scripts\python score.py --input outputs/example_features.csv --output outputs/new_scores.csv
```

The CSV must have the ordered header `x0,x1,...,x19` (all 20 names written out) and
finite numeric values. These anonymous features must match the training feature
meaning/order/distribution; arbitrary business data is not compatible. The sample
file has the correct format. For real data, adapt training and preprocessing first.
Only load trusted joblib models; deserialization can execute code. Retrain locally
if your scikit-learn version differs from the supplied model's environment.

## ECE definition

This project measures positive-class calibration with equal-width bins:
`ECE = sum((n_bin / N) * abs(mean_probability - positive_fraction))`.
Probability 1 belongs in the final bin; empty bins are ignored. ECE is sensitive to
bin count, so we also report 5- and 20-bin results. Neither low ECE nor high ROC-AUC
alone proves a reliable model. See the report for interpretation and limitations.

## Assignment submission

Submit the project ZIP through a shareable Drive or GitHub link, following your
course's instructions. For a walkthrough, show the HTML report, run `train.py`,
run `score.py`, then show the tests. This project is an educational simulation,
not a deployed healthcare, fraud or customer-risk model.
