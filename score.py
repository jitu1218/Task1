"""Score a numeric CSV using the validation-selected fitted model."""
import argparse
import joblib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='outputs/scorer.joblib')
    parser.add_argument('--input', default='outputs/example_features.csv')
    parser.add_argument('--output', default='outputs/new_scores.csv')
    args = parser.parse_args()
    bundle = joblib.load(args.model)  # Only load model files from trusted sources.
    expected = ','.join(f'x{i}' for i in range(bundle['feature_count']))
    with open(args.input, encoding='utf-8') as source:
        if source.readline().strip() != expected:
            raise ValueError('Expected ordered CSV header: ' + expected)
    X = np.loadtxt(args.input, delimiter=',', skiprows=1, ndmin=2)
    if X.shape[1] != bundle['feature_count'] or not len(X) or not np.isfinite(X).all():
        raise ValueError('Expected finite numeric features with 20 columns')
    p = bundle['model'].predict_proba(X)[:, 1]
    np.savetxt(args.output, p[:, None], delimiter=',', header='probability_class_1', comments='')
    print('Scorer:', bundle['name'])
    print('Probabilities:', np.round(p, 4))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
