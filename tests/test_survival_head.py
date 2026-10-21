import torch

from libs.modeling.survival import SurvivalHead


def test_survival_head_shapes_and_gradients():
    head = SurvivalHead(embd_dim=8, hidden_dim=16, n_layers=2, dropout=0.0)
    features = (torch.randn(2, 1, 8, 12, requires_grad=True),)
    masks = (torch.ones(2, 1, 1, 12, dtype=torch.bool),)
    output = head(features, masks)
    assert output['start_logits'].shape == (2, 1, 12)
    assert output['end_logits'].shape == (2, 1, 12)
    assert output['cure_logits'].shape == (2, 1, 12)
    assert output['state_logits'].shape == (2, 1, 12, 3)
    sum(value.float().sum() for key, value in output.items() if key != 'mask').backward()
    assert features[0].grad is not None
    assert torch.isfinite(features[0].grad).all()


def test_survival_head_does_not_use_future_features():
    torch.manual_seed(3)
    head = SurvivalHead(embd_dim=8, hidden_dim=16, n_layers=2, dropout=0.0).eval()
    first = torch.randn(1, 1, 8, 24)
    second = first.clone()
    cutoff = 11
    second[..., cutoff + 1:] = torch.randn_like(second[..., cutoff + 1:]) * 50
    mask = (torch.ones(1, 1, 1, 24, dtype=torch.bool),)
    with torch.no_grad():
        out_first = head((first,), mask)
        out_second = head((second,), mask)
    for key in ('start_logits', 'end_logits', 'cure_logits'):
        torch.testing.assert_close(
            out_first[key][..., :cutoff + 1],
            out_second[key][..., :cutoff + 1],
        )
    torch.testing.assert_close(
        out_first['state_logits'][..., :cutoff + 1, :],
        out_second['state_logits'][..., :cutoff + 1, :],
    )


def test_paper_decoder_leakage_for_both_fp_settings():
    # The decoder consumes the same F0 interface independently of HEM's FP branch.
    # Check every added output, including the intermediate d=1 and d=2 blocks.
    from pathlib import Path
    from omegaconf import OmegaConf
    from libs.config import load_config
    from libs.modeling.survival import make_survival_head
    root = Path(__file__).resolve().parents[1]
    for future_prediction in (False, True):
        cfg = load_config(dict(config=str(root/'configs/anet.yaml'),
                               model=dict(future_prediction=future_prediction)))
        head = make_survival_head(OmegaConf.to_object(cfg.model.survival_head)).eval()
        first = torch.randn(1,1,128,24)
        second = first.clone()
        second[...,12:] = torch.randn_like(second[...,12:])*50
        mask = torch.ones(1,1,1,24,dtype=torch.bool)
        with torch.no_grad():
            left = head((first,),(mask,))
            right = head((second,),(mask,))
            for key in ('start_logits','end_logits','cure_logits'):
                torch.testing.assert_close(left[key][...,:12],right[key][...,:12])
            torch.testing.assert_close(left['state_logits'][...,:12,:],right['state_logits'][...,:12,:])
            x,y = first.reshape(1,128,24),second.reshape(1,128,24)
            m = mask.reshape(1,1,24)
            for conv,norm in zip(head.convs,head.norms):
                x,_ = conv(x,m)
                y,_ = conv(y,m)
                x,y = torch.nn.functional.gelu(norm(x)),torch.nn.functional.gelu(norm(y))
                torch.testing.assert_close(x[...,:12],y[...,:12])
