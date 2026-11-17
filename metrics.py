"""Positive-class, equal-width expected calibration error."""
import numpy as np


def reliability(y, probabilities, bins=10):
    y, p = np.asarray(y), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError('Nonempty matching one-dimensional arrays required')
    if not np.all(np.isin(y, [0, 1])) or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('Binary labels and finite probabilities in [0, 1] required')
    if type(bins) is not int or bins < 1:
        raise ValueError('Positive integer bins required')
    indices = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for i in range(bins):
        selected = indices == i
        count = int(selected.sum())
        rows.append({'bin': i, 'lower': i / bins, 'upper': (i + 1) / bins,
                     'count': count,
                     'mean_probability': float(p[selected].mean()) if count else None,
                     'observed_frequency': float(y[selected].mean()) if count else None})
    return rows


def ece(y, probabilities, bins=10):
    rows = reliability(y, probabilities, bins)
    return sum(r['count'] * abs(r['mean_probability'] - r['observed_frequency'])
               for r in rows if r['count']) / len(y)
