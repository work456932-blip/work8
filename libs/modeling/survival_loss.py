"""Composite survival/Direct-MT objectives, matching Sec. 3 and Supp. A.

Continuous feature-grid boundaries use ceil(): an event enters the discrete
process at its first visible grid position. Empty risk sets contribute zero.
The inherited per-risk-position reduction is an explicit implementation default.
"""
import torch
import torch.nn.functional as F


def _masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype)
    return torch.where(mask.bool(), values, torch.zeros_like(values)).sum() / mask.sum().clamp(min=1.0)


def discrete_event_nll(logits, mask, event_time, observed, entry_time=None,
                       include=None, normalize_by_risk=True):
    logits = logits.float()
    grid = torch.arange(logits.size(-1), device=logits.device)[None]
    event = torch.ceil(event_time.float()).long()
    entry = torch.zeros_like(event) if entry_time is None else torch.ceil(entry_time).long().clamp(min=0)
    include = torch.ones_like(observed, dtype=torch.bool) if include is None else include.bool()
    survival = mask.bool() & (grid >= entry[:, None]) & include[:, None]
    survival &= (~observed[:, None]) | (grid < event[:, None])
    hit = mask.bool() & include[:, None] & observed[:, None] & (grid == event[:, None]) & (grid >= entry[:, None])
    loss = (F.softplus(logits) * survival).sum(-1) + (F.softplus(-logits) * hit).sum(-1)
    if normalize_by_risk:
        loss = loss / (survival.sum(-1) + hit.sum(-1)).clamp(min=1)
    return _masked_mean(loss, include)


def direct_boundary_log_probs(logits, mask):
    """Supp. Eq. 1: observed positions + a fixed-zero out-of-prefix logit.

    The zero reference is a parameter-free softmax convention, so Direct-MT
    and SurvVTG instantiate exactly the same decoder and four linear heads.
    Only the boundary objective/normalization changes.
    """
    scores = logits.float().masked_fill(~mask.bool(), float('-inf'))
    outside = scores.new_zeros((*scores.shape[:-1], 1))
    return F.log_softmax(torch.cat((scores, outside), dim=-1), dim=-1)


def direct_boundary_loss(logits, mask, event_time, relevant):
    length = mask.sum(-1)
    index = torch.ceil(event_time).long()
    include = ((~relevant) | (event_time >= 0)) & (length > 0)
    visible = relevant & (event_time >= 0) & (index < length)
    targets = torch.where(visible, index, torch.full_like(index, logits.size(-1)))
    loss = -direct_boundary_log_probs(logits, mask).gather(-1, targets.clamp(min=0)[:, None]).squeeze(-1)
    return _masked_mean(loss, include)


def event_state_targets(start_time, end_time, relevant, seq_len, device):
    grid = torch.arange(seq_len, device=device)[None]
    labels = torch.zeros((len(start_time), seq_len), device=device, dtype=torch.long)
    labels = torch.where(grid >= start_time[:, None], 1, labels)
    labels = torch.where(grid >= end_time[:, None], 2, labels)
    return labels, relevant[:, None].bool()


def multistate_survival_loss(outputs, target_segments, relevant, weights=None,
                              normalize_by_risk=True, boundary_objective='hazard'):
    w = dict(start=1., end=1., cure=.5, state=.5, monotonic=.05)
    w.update(weights or {})
    start_logits = outputs['start_logits'].flatten(0, 1)
    end_logits = outputs['end_logits'].flatten(0, 1)
    mask = outputs['mask'].flatten(0, 1).bool()
    target = target_segments.reshape(-1, 2).float()
    relevant = relevant.flatten().bool()
    if len(start_logits) != len(target):
        raise ValueError('Survival outputs and targets must have identical query/window order')
    start, end = target.unbind(-1)
    length = mask.sum(-1)
    valid = length > 0
    start_visible = relevant & (start >= 0) & (torch.ceil(start) < length)
    end_visible = relevant & (end >= 0) & (torch.ceil(end) < length)
    start_include = ((~relevant) | (start >= 0)) & valid
    # LC start: enter the end process at the crop origin if still active there.
    end_include = relevant & (torch.ceil(start) < length) & (end >= 0) & valid
    if boundary_objective == 'hazard':
        ls = discrete_event_nll(start_logits, mask, start, start_visible,
                                include=start_include, normalize_by_risk=normalize_by_risk)
        le = discrete_event_nll(end_logits, mask, end, end_visible,
                                entry_time=start.clamp(min=0), include=end_include,
                                normalize_by_risk=normalize_by_risk)
    elif boundary_objective == 'direct':
        ls = direct_boundary_loss(start_logits, mask, start, relevant)
        le = direct_boundary_loss(end_logits, mask, end, relevant)
    else:
        raise ValueError('boundary_objective must be hazard or direct')
    zero = start_logits.sum() * 0.
    lc = lz = lm = zero
    if outputs.get('cure_logits') is not None:
        cure = outputs['cure_logits'].flatten(0, 1).float()
        raw = F.binary_cross_entropy_with_logits(cure, (~relevant)[:, None].float().expand_as(cure), reduction='none')
        lc = _masked_mean(raw, mask)
    if outputs.get('state_logits') is not None:
        state = outputs['state_logits'].flatten(0, 1).float()
        labels, state_valid = event_state_targets(start, end, relevant, state.size(1), state.device)
        raw = F.cross_entropy(state.reshape(-1, 3), labels.flatten(), reduction='none').reshape_as(labels)
        lz = _masked_mean(raw, mask & state_valid)
        prob = state.softmax(-1)
        pairs = mask[:, 1:] & mask[:, :-1] & relevant[:, None]
        lm = _masked_mean(F.relu(prob[:, 1:, 0] - prob[:, :-1, 0]) + F.relu(prob[:, :-1, 2] - prob[:, 1:, 2]), pairs)
    return dict(surv_start=ls, surv_end=le, surv_cure=lc, surv_state=lz,
                surv_monotonic=lm, survival=w['start']*ls+w['end']*le+w['cure']*lc+w['state']*lz+w['monotonic']*lm)


@torch.no_grad()
def survival_curves(outputs, boundary_objective='hazard'):
    """Interpret Direct-MT as categorical masses, never as sigmoid hazards."""
    mask = outputs['mask'].bool()
    result = dict(mask=mask, boundary_objective=boundary_objective)
    for boundary in ('start', 'end'):
        logits = outputs[boundary + '_logits'].float()
        if boundary_objective == 'direct':
            prob = direct_boundary_log_probs(logits, mask).exp()
            mass = prob[..., :-1]
            survival = (1. - mass.cumsum(-1)).clamp(0., 1.)
            previous = torch.cat((torch.ones_like(survival[..., :1]), survival[..., :-1]), -1)
            hazard = mass / previous.clamp(min=1e-8)
            result[boundary + '_outside'] = prob[..., -1]
            result[boundary + '_mass'] = mass
        else:
            hazard = logits.sigmoid() * mask
            survival = torch.cumprod(1. - hazard, dim=-1)
        result[boundary + '_hazard'] = hazard
        result[boundary + '_survival'] = survival
    cure = outputs.get('cure_logits')
    result['cure'] = cure.float().sigmoid() if cure is not None else None
    result['state'] = outputs['state_logits'].float().softmax(-1) if outputs.get('state_logits') is not None else None
    result['mixture_start_survival'] = result['start_survival'] if cure is None else result['cure'] + (1. - result['cure']) * result['start_survival']
    return result
