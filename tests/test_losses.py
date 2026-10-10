import torch

from vjev.losses import typed_loss


def test_confident_correct_is_low():
    logits = torch.tensor([10.0, 0.0, 0.0])
    out = typed_loss(logits, 0, smoothing=0.0)
    assert out["brier"] < 1e-6
    assert out["emd"] < 1e-6
    assert out["loss"] < 1e-3


def test_emd_prefers_near_miss():
    near = typed_loss(torch.tensor([0.0, 10.0, 0.0, 0.0]), 0)["emd"]
    far = typed_loss(torch.tensor([0.0, 0.0, 0.0, 10.0]), 0)["emd"]
    assert near < far


def test_gradient_flows():
    logits = torch.zeros(5, requires_grad=True)
    typed_loss(logits, 2)["loss"].backward()
    assert logits.grad is not None
    assert logits.grad[2] < 0  # raising the target logit lowers the loss


def test_soft_target_matches_onehot_and_prefers_mass_inside():
    logits = torch.tensor([0.0, 2.0, 0.0])
    a = typed_loss(logits, 1)["loss"]
    b = typed_loss(logits, [0.0, 1.0, 0.0])["loss"]
    assert torch.allclose(a, b)
    inside = typed_loss(torch.tensor([0.0, 3.0, 3.0, 0.0]), [0, 0.5, 0.5, 0])["loss"]
    outside = typed_loss(torch.tensor([3.0, 0.0, 0.0, 3.0]), [0, 0.5, 0.5, 0])["loss"]
    assert inside < outside
