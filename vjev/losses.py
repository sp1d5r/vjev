"""Training loss for one typed question, following the Clef recipe.

loss = label-smoothed cross-entropy + brier_w * Brier + emd_w * ordinal EMD.
Brier pushes probabilities toward calibration.
EMD (squared distance between cumulative distributions) gives partial credit to
near-miss levels of a `score` question.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def typed_loss(logits: torch.Tensor, target: int | list[float], smoothing: float = 0.05,
               brier_w: float = 1.0, emd_w: float = 1.0, ordinal: bool = True) -> dict:
    """target is a level index, or a distribution over options (for `moment` questions)."""
    logits = logits.float()
    k = logits.shape[-1]
    if not isinstance(target, (list, tuple)):
        dist = F.one_hot(torch.tensor(int(target), device=logits.device), k).float()
    else:
        dist = torch.tensor(target, device=logits.device, dtype=torch.float32)
    smoothed = dist * (1 - smoothing) + smoothing / k
    ce = -(smoothed * logits.log_softmax(-1)).sum()
    probs = logits.softmax(-1)
    brier = ((probs - dist) ** 2).sum()
    emd = ((probs.cumsum(-1) - dist.cumsum(-1)) ** 2).sum() if ordinal else probs.new_zeros(())
    total = ce + brier_w * brier + emd_w * emd
    return {"loss": total, "ce": ce.detach(), "brier": brier.detach(), "emd": emd.detach()}
