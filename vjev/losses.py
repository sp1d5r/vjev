"""Training loss for one typed question, following the Clef recipe.

loss = label-smoothed cross-entropy + brier_w * Brier + emd_w * ordinal EMD.
Brier pushes probabilities toward calibration.
EMD (squared distance between cumulative distributions) gives partial credit to
near-miss levels of a `score` question.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def typed_loss(logits: torch.Tensor, target: int, smoothing: float = 0.05,
               brier_w: float = 1.0, emd_w: float = 1.0, ordinal: bool = True) -> dict:
    logits = logits.float()
    k = logits.shape[-1]
    target_t = torch.tensor(target, device=logits.device)
    ce = F.cross_entropy(logits.unsqueeze(0), target_t.unsqueeze(0), label_smoothing=smoothing)
    probs = logits.softmax(-1)
    onehot = F.one_hot(target_t, k).float()
    brier = ((probs - onehot) ** 2).sum()
    emd = ((probs.cumsum(-1) - onehot.cumsum(-1)) ** 2).sum() if ordinal else probs.new_zeros(())
    total = ce + brier_w * brier + emd_w * emd
    return {"loss": total, "ce": ce.detach(), "brier": brier.detach(), "emd": emd.detach()}
