import torch

from libs.modeling.optim import make_optimizer
from libs.modeling.survival import SurvivalHead


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vid_net = torch.nn.Linear(2, 2)
        self.text_net = torch.nn.Linear(2, 2)
        self.fusion = torch.nn.Linear(2, 2)
        self.cls_head = torch.nn.Linear(2, 1)
        self.reg_head = torch.nn.Linear(2, 2)
        self.survival_head = SurvivalHead(2, hidden_dim=4, n_layers=1)


def test_optimizer_contains_survival_parameters():
    model = TinyModel()
    optimizer = make_optimizer(
        model, {'name': 'adamw', 'lr': 1e-3, 'weight_decay': 0.01}
    )
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
    assert all(id(parameter) in optimized for parameter in model.survival_head.parameters())
