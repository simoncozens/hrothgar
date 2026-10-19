"""Behavioural tests for the instance-membership model and loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hrothgar.set_membership.config import SetMembershipConfig
from hrothgar.set_membership.evaluation import _rank_metrics
from hrothgar.set_membership.losses import glyph_contrastive_loss
from hrothgar.set_membership.model import SetMembershipModel


def _clustered(n_instances: int = 4, per: int = 6, dim: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.eye(n_instances, dim) * 5.0
    z, ids = [], []
    for i in range(n_instances):
        for _ in range(per):
            z.append(
                F.normalize(centres[i] + torch.randn(dim, generator=g) * 0.05, dim=-1)
            )
            ids.append(i)
    return torch.stack(z), torch.tensor(ids)


def test_glyph_contrastive_loss_prefers_correct_ids() -> None:
    z, ids = _clustered()
    correct = glyph_contrastive_loss(z, ids)
    assert torch.isfinite(correct)

    scrambled = glyph_contrastive_loss(z, ids[torch.randperm(ids.numel())])
    assert correct < scrambled


def test_glyph_contrastive_loss_is_zero_without_positives() -> None:
    # One glyph per instance -> no positive pairs -> loss is zero, not NaN.
    z = F.normalize(torch.randn(3, 8), dim=-1)
    loss = glyph_contrastive_loss(z, torch.tensor([0, 1, 2]))
    assert torch.isfinite(loss) and loss == 0.0


def test_model_forward_is_normalized() -> None:
    config = SetMembershipConfig()
    model = SetMembershipModel(config).eval()
    with torch.no_grad():
        z = model(torch.randn(4, 1, config.glyph_size, config.glyph_size))
    assert z.shape == (4, config.embedding_dim)
    assert (z.norm(dim=-1) - 1).abs().max() < 1e-3


def test_model_is_image_only() -> None:
    # Geometry is kept in the pipeline for revert but the model ignores it.
    config = SetMembershipConfig()
    model = SetMembershipModel(config).eval()
    assert not hasattr(model, "geometry_mlp")
    with torch.no_grad():
        z = model(torch.randn(3, 1, 128, 128))
    assert z.shape == (3, config.embedding_dim)


def test_prototype_is_normalized_mean() -> None:
    z = torch.randn(6, 16)
    prototype = SetMembershipModel.prototype(z)
    assert prototype.shape == (16,)
    assert abs(prototype.norm() - 1) < 1e-5


def test_rank_metrics_scores_clustered_candidates() -> None:
    g = torch.Generator().manual_seed(1)
    candidates = F.normalize(
        torch.eye(4, 16) * 5.0 + torch.randn(4, 16, generator=g) * 0.01, dim=-1
    )
    query = F.normalize(candidates + torch.randn(4, 16, generator=g) * 0.001, dim=-1)
    metric = _rank_metrics(query, candidates, torch.tensor([0, 1, 2, 3]))
    assert metric["r1"] == 1.0
    assert metric["auc"] == 1.0
    assert metric["n"] == 4
