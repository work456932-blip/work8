import torch
from libs.modeling.survival_loss import direct_boundary_loss, multistate_survival_loss, survival_curves


def test_direct_ce_visible_future_irrelevant_and_lc():
    logits = torch.zeros(4,4,requires_grad=True)
    mask = torch.ones_like(logits,dtype=torch.bool)
    time = torch.tensor([2.,8.,-1.,-2.])
    relevant = torch.tensor([True,True,False,True])
    loss = direct_boundary_loss(logits,mask,time,relevant)
    torch.testing.assert_close(loss,torch.log(torch.tensor(5.)))
    loss.backward()
    assert logits.grad[0,2] < 0
    assert (logits.grad[1] > 0).all() and (logits.grad[2] > 0).all()
    assert (logits.grad[3] == 0).all()


def test_end_hazard_has_no_pre_event_or_irrelevant_gradient():
    outputs = dict(start_logits=torch.zeros(2,1,4,requires_grad=True),
                   end_logits=torch.zeros(2,1,4,requires_grad=True),
                   mask=torch.ones(2,1,4,dtype=torch.bool))
    loss = multistate_survival_loss(outputs,torch.tensor([[8.,10.],[-1.,-1.]]),torch.tensor([True,False]))
    loss['surv_end'].backward()
    assert (outputs['end_logits'].grad == 0).all()


def test_direct_interpretation_is_categorical():
    output = dict(start_logits=torch.zeros(1,1,4),end_logits=torch.zeros(1,1,4),mask=torch.ones(1,1,4,dtype=torch.bool))
    curves = survival_curves(output,'direct')
    torch.testing.assert_close(curves['start_survival'][0,0],torch.tensor([.8,.6,.4,.2]))
