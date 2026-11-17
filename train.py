"""Leakage-free training, calibration, model selection and final evaluation."""
import argparse
import csv
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from metrics import ece, reliability


def split_indices(y, seed=42):
    indices = np.arange(len(y))
    train, rest = train_test_split(indices, test_size=.5, stratify=y, random_state=seed)
    cal, rest = train_test_split(rest, train_size=.4, stratify=y[rest], random_state=seed + 1)
    validation, test = train_test_split(rest, test_size=.5, stratify=y[rest], random_state=seed + 2)
    return {'train': train, 'calibration': cal, 'validation': validation, 'test': test}


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(name, y, p):
    return {'model': name, 'brier': brier_score_loss(y, p), 'log_loss': log_loss(y, p),
            'ece_10': ece(y, p), 'ece_5': ece(y, p, 5), 'ece_20': ece(y, p, 20),
            'roc_auc': roc_auc_score(y, p)}


def plot_results(out, y, predictions, results):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#64748b', '#007f86', '#db7c26']
    methods = ['raw', 'sigmoid', 'isotonic']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    for col, family in enumerate(['Logistic regression', 'Random forest']):
        ax = axes[0, col]
        ax.plot([0, 1], [0, 1], '--', color='#a8a8a8', label='Perfect calibration')
        for method, color in zip(methods, colors):
            p = predictions[f'{family} | {method}']
            rows = [r for r in reliability(y, p) if r['count']]
            ax.plot([r['mean_probability'] for r in rows],
                    [r['observed_frequency'] for r in rows], 'o-', color=color, label=method)
            axes[1, col].hist(p, bins=np.linspace(0, 1, 11), histtype='step',
                              linewidth=2, color=color, label=method)
        ax.set(title=family, xlabel='Mean predicted P(class 1)', ylabel='Observed positive frequency',
               xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=9)
        ax.grid(alpha=.15)
        axes[1, col].set(xlabel='Predicted P(class 1)', ylabel='Test records per bin', xlim=(0, 1))
        axes[1, col].legend()
    fig.suptitle('Do probability scores match observed outcomes?\nUntouched test set · 10 equal-width bins', fontsize=17)
    fig.savefig(out / 'reliability_diagrams.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout='constrained')
    labels = [r['model'].replace('Logistic regression', 'LR').replace('Random forest', 'RF') for r in results]
    for ax, metric, title in zip(axes, ['ece_10', 'brier', 'log_loss'], ['ECE (10 bins)', 'Brier score', 'Log loss']):
        ax.barh(labels, [r[metric] for r in results], color=colors * 2)
        ax.invert_yaxis()
        ax.set_title(title + ' — lower is better')
        ax.grid(axis='x', alpha=.15)
    fig.savefig(out / 'metric_comparison.png', dpi=170)
    plt.close(fig)


def report(out, results, chosen, validation, splits, y, seed):
    table = '| Model | Brier ↓ | Log loss ↓ | ECE-10 ↓ | ROC-AUC ↑ |\n|---|---:|---:|---:|---:|\n'
    for r in results:
        table += f"| {r['model'].replace(' | ', ' / ')} | {r['brier']:.4f} | {r['log_loss']:.4f} | {r['ece_10']:.4f} | {r['roc_auc']:.4f} |\n"
    findings = []
    for family in ['Logistic regression', 'Random forest']:
        baseline = next(r for r in results if r['model'] == family + ' | raw')
        for method in ['sigmoid', 'isotonic']:
            row = next(r for r in results if r['model'] == family + ' | ' + method)
            findings.append(f"- {family}, {method}: ECE changed from {baseline['ece_10']:.4f} to {row['ece_10']:.4f}; "
                            f"Brier changed from {baseline['brier']:.4f} to {row['brier']:.4f}.")
    text = f'''# Calibrated Probability Scoring — Experiment Report

## Objective and data
Compare probability quality before and after calibration. The reproducible synthetic
binary dataset has {len(y):,} rows, 20 numeric features (10 informative, 5 redundant),
nominal 70/30 class weighting, and 3% random label reassignment. Observed positive
prevalence is {y.mean():.2%}. Seed: {seed}. It illustrates a risk/churn-style scoring
task; these are not real customers or a validated domain model. Positive class is 1.

## Method and leakage prevention
Stratified disjoint splits: {len(splits['train'])} training, {len(splits['calibration'])}
calibration, {len(splits['validation'])} validation, {len(splits['test'])} final test.
Logistic regression uses a scaler fitted only on training data. Random forest uses
200 trees, minimum leaf size 5. Hyperparameters are fixed in advance. Each fitted
base model is frozen; sigmoid (Platt-style) and isotonic mappings learn exclusively
from the calibration split. Six candidates share the same held-out sets.
The saved scorer **{chosen}** was selected by lowest validation Brier score
({min(r['brier'] for r in validation):.4f}), before test evaluation. Models are not
refitted after selection. Test comparisons describe results, not a new selection rule.

## Final test results
{table}
{chr(10).join(findings)}

## Visual evidence
![Reliability diagrams and bin support](reliability_diagrams.png)
![Metric comparison](metric_comparison.png)

The diagonal represents agreement between mean predictions and observed frequencies.
Points above it indicate underprediction of positive probability; points below it
indicate overprediction. Histograms show support: sparse bins are less reliable.
Empty bins are omitted from curves and contribute zero to ECE. Detailed bin counts
are in reliability_bins.csv; per-row probabilities are in test_predictions.csv.

## Metrics and interpretation
ECE = sum over bins of (bin count / N) × |observed positive fraction − mean predicted
probability|. Bins are [0, .1), [.1, .2), …, [.9, 1], including probability 1 in the
last bin. This is positive-class ECE, not top-label confidence ECE. Lower is better.
ECE depends on binning and sample size; metrics.csv also includes 5- and 20-bin ECE.
Brier is mean squared probability error. Log loss strongly penalizes confident wrong
predictions. Both assess probability quality, including discrimination, rather than
calibration alone. ROC-AUC assesses ranking, not probability accuracy.

Calibration does not guarantee improvement for every model or metric. Sigmoid learns
a smooth mapping; isotonic learns a flexible monotone mapping and can overfit or
produce extreme 0/1 estimates. Isotonic may change AUC through tied scores. A model
can rank cases well yet attach inaccurate probabilities to them. Use all diagnostics,
not just a single low ECE or an attractive curve.

## Decision-making example
If false positives cost 1 unit and false negatives cost 4 units, the minimum expected
cost decision under that simple two-error-cost model uses threshold 1/(1+4) = 0.20.
At p=0.70, the predicted positive outcome should occur around 70% of the time among
comparable cases in aggregate, not with certainty for one individual. Calibration
therefore supports expected-cost decisions, risk estimates and resource planning.
The costs here are illustrative, not business advice or an optimized test threshold.

## Limitations and next steps
One synthetic dataset and one split do not establish statistical significance or
real-world readiness. No confidence intervals are claimed. In deployment, use
representative data, validate temporal/group splits where appropriate, inspect
subgroup calibration, monitor prevalence/drift, and recalibrate when necessary.
Use repeated validation or a fresh holdout before making stronger comparative claims.

## References
- Scikit-learn probability calibration guide: https://scikit-learn.org/stable/modules/calibration.html
- CalibratedClassifierCV: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
'''
    (out / 'REPORT.md').write_text(text, encoding='utf-8')
    # Standalone browser-readable report, with the figures embedded for portability.
    import base64
    import html
    parts = ['<!doctype html><meta charset="utf-8"><title>Probability Calibration Report</title>',
             '<style>body{max-width:1080px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#173247}img{width:100%;margin:20px 0}h1,h2{color:#007f86}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}th{background:#e8f4f5}</style>']
    table_open = False
    for line in text.splitlines():
        if line.startswith('|'):
            if line.startswith('|---'):
                continue
            tag = 'td' if table_open else 'th'
            if not table_open:
                parts.append('<table>')
                table_open = True
            cells = line.strip('|').split('|')
            # Model names themselves contain a separator; merge their first two cells.
            if tag == 'td' and len(cells) == 6:
                cells = [cells[0] + ' / ' + cells[1]] + cells[2:]
            parts.append('<tr>' + ''.join(f'<{tag}>' + html.escape(c.strip()) + f'</{tag}>' for c in cells) + '</tr>')
            continue
        if table_open:
            parts.append('</table>')
            table_open = False
        if line.startswith('!['):
            name = line.split('](')[1][:-1]
            encoded = base64.b64encode((out / name).read_bytes()).decode()
            parts.append(f'<img alt="{name}" src="data:image/png;base64,{encoded}">')
        elif line.startswith('## '):
            parts.append('<h2>' + html.escape(line[3:]) + '</h2>')
        elif line.startswith('# '):
            parts.append('<h1>' + html.escape(line[2:]) + '</h1>')
        elif not line:
            parts.append('<br>')
        else:
            parts.append(html.escape(line).replace('**', '') + '\n')
    (out / 'REPORT.html').write_text('\n'.join(parts), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=10,
        n_redundant=5, weights=[.7, .3], flip_y=.03, class_sep=1, random_state=args.seed)
    splits = split_indices(y, args.seed)
    tr, ca, va, te = [splits[k] for k in ['train', 'calibration', 'validation', 'test']]
    bases = {'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
             'Random forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                     random_state=args.seed, n_jobs=-1)}
    candidates = {}
    for name, estimator in bases.items():
        estimator.fit(X[tr], y[tr])
        candidates[name + ' | raw'] = estimator
        for method in ['sigmoid', 'isotonic']:
            model = CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
            model.fit(X[ca], y[ca])
            candidates[name + ' | ' + method] = model
    validation = [evaluate(name, y[va], model.predict_proba(X[va])[:, 1]) for name, model in candidates.items()]
    chosen = min(validation, key=lambda r: r['brier'])['model']
    predictions = {name: model.predict_proba(X[te])[:, 1] for name, model in candidates.items()}
    results = [evaluate(name, y[te], p) for name, p in predictions.items()]
    csv_write(out / 'metrics.csv', results)
    csv_write(out / 'validation_metrics.csv', validation)
    csv_write(out / 'test_predictions.csv', [dict(sample_id=int(i), actual=int(y[i]),
        **{name: float(p[j]) for name, p in predictions.items()}) for j, i in enumerate(te)])
    csv_write(out / 'reliability_bins.csv', [dict(model=name, **r) for name, p in predictions.items()
                                             for r in reliability(y[te], p)])
    manifest = {'seed': args.seed, 'selected_model': chosen, 'selection_metric': 'validation Brier',
                'splits': {k: v.tolist() for k, v in splits.items()}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    joblib.dump({'model': candidates[chosen], 'name': chosen, 'feature_count': 20}, out / 'scorer.joblib')
    np.savetxt(out / 'example_features.csv', X[te[:5]], delimiter=',',
               header=','.join(f'x{i}' for i in range(20)), comments='')
    plot_results(out, y[te], predictions, results)
    report(out, results, chosen, validation, splits, y, args.seed)
    print('Selected using validation:', chosen)
    for r in results:
        print(f"{r['model']:32s} ECE={r['ece_10']:.4f} Brier={r['brier']:.4f} AUC={r['roc_auc']:.4f}")
    print('Report:', out / 'REPORT.html')


if __name__ == '__main__':
    main()
