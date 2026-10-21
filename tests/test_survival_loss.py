import torch

from libs.modeling.survival_loss import discrete_event_nll, multistate_survival_loss


def test_observed_event_prefers_high_hazard_at_event_time():
    mask = torch.ones(1, 6, dtype=torch.bool)
    event_time = torch.tensor([3.0])
    observed = torch.tensor([True])
    good = torch.full((1, 6), -5.0)
    good[0, 3] = 5.0
    bad = torch.full((1, 6), -5.0)
    bad[0, 3] = -8.0
    assert discrete_event_nll(good, mask, event_time, observed) < discrete_event_nll(
        bad, mask, event_time, observed
    )


def test_censored_event_prefers_low_hazards():
    mask = torch.ones(1, 8, dtype=torch.bool)
    observed = torch.tensor([False])
    event_time = torch.tensor([20.0])
    low = torch.full((1, 8), -5.0)
    high = torch.full((1, 8), 2.0)
    assert discrete_event_nll(low, mask, event_time, observed) < discrete_event_nll(
        high, mask, event_time, observed
    )


def test_joint_loss_handles_positive_and_cured_samples():
    start = torch.zeros(2, 1, 10, requires_grad=True)
    end = torch.zeros(2, 1, 10, requires_grad=True)
    cure = torch.zeros(2, 1, 10, requires_grad=True)
    state = torch.zeros(2, 1, 10, 3, requires_grad=True)
    outputs = {
        'start_logits': start,
        'end_logits': end,
        'cure_logits': cure,
        'state_logits': state,
        'mask': torch.ones(2, 1, 10, dtype=torch.bool),
    }
    targets = torch.tensor([[2.0, 6.0], [-1.0, -1.0]])
    relevant = torch.tensor([True, False])
    losses = multistate_survival_loss(outputs, targets, relevant)
    assert all(torch.isfinite(value) for value in losses.values())
    losses['survival'].backward()
    assert start.grad is not None and cure.grad is not None and state.grad is not None
