"""Prefix reliability with valid risk sets and fixed equal-width ECE bins.

Joint NLL excludes cure BCE: start + conditional end, then dataset average.
IBS averages valid temporal positions within each query, then queries.
"""
import math
import numpy as np

EPS = 1e-8


def _binary_auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    keep = np.isfinite(scores)
    labels, scores = labels[keep], scores[keep]
    npos, nneg = (labels == 1).sum(), (labels == 0).sum()
    if not npos or not nneg:
        return float('nan')
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, len(scores)]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[lo:hi]] = (lo + 1 + hi) / 2
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _event_nll(hazard, event_time, observed, entry=0, normalize_by_risk=True):
    hazard = np.clip(hazard, EPS, 1. - EPS)
    entry, index = max(int(math.ceil(entry)), 0), int(math.ceil(event_time))
    if observed:
        before = hazard[entry:index]
        loss = -np.log1p(-before).sum() - np.log(hazard[index])
        count = len(before) + 1
    else:
        before = hazard[entry:]
        loss, count = -np.log1p(-before).sum(), len(before)
    return float(loss / max(count, 1) if normalize_by_risk else loss)


def _bin_stats(scores, targets, edges):
    scores, targets = np.asarray(scores), np.asarray(targets)
    ids = np.clip(np.searchsorted(edges, scores, side='right') - 1, 0, len(edges)-2)
    return dict(count=np.bincount(ids, minlength=len(edges)-1).tolist(),
                confidence=np.bincount(ids, weights=scores, minlength=len(edges)-1).tolist(),
                target=np.bincount(ids, weights=targets, minlength=len(edges)-1).tolist())


def summarize_survival_query(prediction, target_grid, relevant, ece_edges=None, normalize_by_risk=True):
    length = int(np.asarray(prediction['mask'], dtype=bool).sum())
    if not length:
        return None
    edges = np.linspace(0., 1., 16) if ece_edges is None else np.asarray(ece_edges)
    start, end = map(float, target_grid)
    grid = np.arange(length)
    start_observed = relevant and start >= 0 and math.ceil(start) < length
    end_observed = relevant and end >= 0 and math.ceil(end) < length
    start_include = not relevant or start >= 0
    end_include = relevant and math.ceil(start) < length and end >= 0
    direct = prediction.get('boundary_objective') == 'direct'
    def nll(boundary, time, observed, entry=0):
        if direct:
            probability = prediction[boundary + '_mass'][int(math.ceil(time))] if observed else prediction[boundary + '_outside']
            if boundary == 'end' and entry > 0:
                idx = int(math.ceil(entry)) - 1
                probability /= max(float(prediction['end_survival'][idx]), EPS)
            return float(-np.log(np.clip(probability, EPS, 1.)))
        return _event_nll(prediction[boundary + '_hazard'][:length], time, observed, entry, normalize_by_risk)
    snll = nll('start', start, start_observed) if start_include else None
    # Diagnostic start/end NLL uses the valid survival risk sets for both methods.
    enll = nll('end', end, end_observed, max(start, 0)) if end_include else None
    cure_curve = prediction.get('cure')
    cure = float(cure_curve[length-1]) if cure_curve is not None else None
    cure_nll = None if cure is None else float(-np.log(np.clip(cure if not relevant else 1.-cure, EPS, 1.)))
    ss = np.asarray(prediction['start_survival'])[:length]
    mixture = ss if cure_curve is None else np.asarray(cure_curve)[:length] + (1.-np.asarray(cure_curve)[:length])*ss
    true_start = np.ones(length) if not relevant else (grid < start).astype(float)
    sbrier = float(((mixture-true_start)**2).mean()) if start_include else None
    ebrier = None
    end_ece = None
    if end_include:
        risk = grid >= max(start, 0)
        if direct:
            # Convert categorical end masses to survival conditional on entry.
            es = np.asarray(prediction['end_survival'])[:length]
            entry = max(int(math.ceil(start)), 0)
            denominator = 1. if entry == 0 else max(float(es[entry-1]), EPS)
            es = es / denominator
        else:
            hazards = np.asarray(prediction['end_hazard'])[:length]
            es = np.ones(length)
            es[risk] = np.cumprod(1.-hazards[risk])
        ebrier = float(((es[risk]-(grid[risk]<end))**2).mean()) if risk.any() else None
        end_ece = _bin_stats(es[risk], (grid[risk]<end).astype(float), edges) if risk.any() else None
    correct = count = 0
    state = prediction.get('state')
    if relevant and state is not None:
        labels = np.where(grid >= end, 2, np.where(grid >= start, 1, 0))
        correct, count = int((np.asarray(state)[:length].argmax(-1)==labels).sum()), length
    return dict(start_nll=snll, end_nll=enll, cure_nll=cure_nll,
                start_brier=sbrier, end_brier=ebrier, state_correct_sum=correct, state_count=count,
                cure_label=int(not relevant), cure_score=cure,
                end_ece=end_ece,
                start_ece=_bin_stats(mixture, true_start, edges) if start_include else None)


def compute_survival_metrics(results):
    summaries = [r['survival_summary'] for r in results if r.get('survival_summary') is not None]
    def mean(key):
        values = [s[key] for s in summaries if s.get(key) is not None]
        return float(np.mean(values)) if values else float('nan')
    joint = [(s['start_nll'] or 0.)+(s['end_nll'] or 0.) for s in summaries if s['start_nll'] is not None or s['end_nll'] is not None]
    def ece(key):
        bins = [s[key] for s in summaries if s.get(key) is not None]
        if not bins:
            return float('nan')
        counts = np.sum([b['count'] for b in bins], axis=0)
        conf = np.sum([b['confidence'] for b in bins], axis=0)
        target = np.sum([b['target'] for b in bins], axis=0)
        return float(np.abs(conf-target).sum() / max(counts.sum(), 1))
    per_query_ibs = [np.mean([s[k] for k in ('start_brier', 'end_brier') if s[k] is not None]) for s in summaries if s['start_brier'] is not None or s['end_brier'] is not None]
    cured = [s for s in summaries if s['cure_score'] is not None]
    count = sum(s['state_count'] for s in summaries)
    return dict(survival_nll=float(np.mean(joint)) if joint else float('nan'),
                start_nll=mean('start_nll'),end_nll=mean('end_nll'),cure_nll=mean('cure_nll'),
                start_brier=mean('start_brier'),end_brier=mean('end_brier'),
                start_ibs=mean('start_brier'),end_ibs=mean('end_brier'),ibs=float(np.mean(per_query_ibs)) if per_query_ibs else float('nan'),
                start_ece=ece('start_ece'),end_ece=ece('end_ece'),
                state_accuracy=sum(s['state_correct_sum'] for s in summaries)/count if count else float('nan'),
                never_event_auroc=_binary_auc([s['cure_label'] for s in cured],[s['cure_score'] for s in cured]),
                num_survival_queries=len(summaries),num_irrelevant_queries=sum(s['cure_label'] for s in summaries))
