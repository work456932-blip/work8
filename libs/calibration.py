"""Validation-only cure operating points; no HI-ActivityNet tuning."""
import json
import numpy as np


def load_calibration(path, variant):
    with open(path) as handle:
        value = json.load(handle)
    source = value.get('source', {})
    if source.get('role') != 'validation' or source.get('benchmark') == 'hi_activitynet':
        raise ValueError('Calibration must come from non-HI validation data')
    if value.get('model_variant') != variant:
        raise ValueError('Calibration was selected for a different model variant')
    threshold = float(value['threshold'])
    edges = np.asarray(value['ece_edges'], dtype=float)
    if not 0 <= threshold <= 1 or len(edges) < 3 or not np.allclose(edges, np.linspace(0., 1., len(edges))):
        raise ValueError('Require a probability threshold and frozen equal-width ECE bins')
    return value


def select_calibration(payload, bins=15):
    if payload.get('benchmark') == 'hi_activitynet':
        raise ValueError('HI-ActivityNet is excluded from threshold selection')
    split = payload.get('split', '').lower()
    if not (split.startswith('val') or split.startswith('dev')) or 'hi' in split:
        raise ValueError('Use a non-HI validation split')
    records = payload['predictions']
    if any(r.get('refused') for r in records):
        raise ValueError('Select calibration from ungated refusal decisions')
    if not records or any(r.get('cure_probability') is None for r in records):
        raise ValueError('Validation predictions must contain terminal cure probabilities')
    score = np.asarray([r['cure_probability'] for r in records], dtype=float)
    positive = np.asarray([not r['is_relevant'] for r in records], dtype=bool)
    if not np.isfinite(score).all() or not positive.any() or positive.all():
        raise ValueError('Need finite scores and both relevant and irrelevant validation pairs')
    # Threshold selection criterion is unspecified in the paper. Use never-event
    # detection F1, deterministic ties favor the larger (less refusing) threshold.
    best = (-1., 1.)
    for threshold in np.unique(np.r_[0., score, 1.]):
        predict = score >= threshold
        tp = (predict & positive).sum()
        fp = (predict & ~positive).sum()
        fn = (~predict & positive).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        best = max(best, (float(f1), float(threshold)))
    return dict(threshold=best[1], ece_edges=np.linspace(0., 1., bins+1).tolist(),
                model_variant=payload['model_variant'], selection='never_event_detection_f1',
                source=dict(role='validation', benchmark=payload.get('benchmark', 'standard'), split=payload['split']))
