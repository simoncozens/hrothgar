"""Behavioural tests for the similarity-retrieval metrics.

Each test plants a known structure (separated clusters, a collapsed line, a
near/far perturbation) and checks the metric reports the structure it is
supposed to report.  These are invariants, not interface checks: a metric that
breaks them is measuring the wrong thing.
"""

from __future__ import annotations

import torch

from hrothgar.style_embedding.metrics import (
    cross_set_purity,
    geometry_stats,
    label_agreement_at_k,
    membership_ranking,
    nearest_neighbours,
    pairing_scores,
    sibling_retrieval,
    signal_attribution,
    similarity_vector,
)


def _clustered(
    n_labels: int = 4,
    per_label: int = 8,
    dim: int = 32,
    spread: float = 0.05,
    seed: int = 0,
) -> tuple[torch.Tensor, list[str]]:
    g = torch.Generator().manual_seed(seed)
    rows: list[torch.Tensor] = []
    labels: list[str] = []
    for c in range(n_labels):
        centre = torch.zeros(dim)
        centre[c] = 10.0
        for _ in range(per_label):
            rows.append(centre + torch.randn(dim, generator=g) * spread)
            labels.append(f"fam{c}")
    return torch.stack(rows), labels


def test_sibling_retrieval_recovers_planted_clusters() -> None:
    embeddings, labels = _clustered()
    score = sibling_retrieval(embeddings, labels)
    assert score.n_queries == len(labels)
    assert score.recall[1] == 1.0
    assert score.mrr == 1.0


def test_sibling_retrieval_ignores_singletons() -> None:
    embeddings, labels = _clustered()
    embeddings = torch.cat([embeddings, torch.randn(1, 32)], dim=0)
    labels = labels + ["only-one"]
    score = sibling_retrieval(embeddings, labels)
    assert score.n_pool == len(labels)
    assert score.n_queries == len(labels) - 1


def test_label_agreement_matches_structure_then_chance() -> None:
    embeddings, labels = _clustered()
    structured = label_agreement_at_k(embeddings, labels)
    assert structured["overall"] > 0.95

    # Same geometry, scrambled labels: neighbours are still cluster-mates, but
    # the labels no longer identify the cluster, so agreement drops to chance.
    g = torch.Generator().manual_seed(1)
    scrambled = [labels[i] for i in torch.randperm(len(labels), generator=g).tolist()]
    assert label_agreement_at_k(embeddings, scrambled)["overall"] < 0.4


def test_geometry_stats_detects_collapse_and_is_isotropic() -> None:
    g = torch.Generator().manual_seed(0)
    direction = torch.randn(64, generator=g)
    collapsed = (
        torch.randn(200, 64, generator=g) * 0.01
        + torch.randn(200, 1, generator=g) * 5.0 * direction
    )
    assert geometry_stats(collapsed)["participation_ratio"] < 2.0

    isotropic = torch.randn(400, 64, generator=g)
    assert geometry_stats(isotropic)["participation_ratio"] > 0.7 * 64


def test_pairing_scores_invariance_and_discrimination() -> None:
    embeddings, _ = _clustered()
    anchors = [0, 8, 16, 24]
    g = torch.Generator().manual_seed(2)

    near = embeddings[anchors] + torch.randn(len(anchors), 32, generator=g) * 0.01
    near_scores = pairing_scores(embeddings, anchors, near)
    assert near_scores["recall"][1] == 1.0
    assert near_scores["cos_same_pair"] > near_scores["cos_unrelated"]
    assert near_scores["discrimination_gap"] > 0

    # A "perturbation" that lands in an unrelated cluster must not retrieve it.
    far = embeddings[[8, 16, 24, 0]]
    assert pairing_scores(embeddings, anchors, far)["recall"][1] == 0.0


def test_pairing_family_group_credits_sibling_match() -> None:
    embeddings, labels = _clustered(n_labels=2, per_label=4)
    g = torch.Generator().manual_seed(4)
    # Anchor is row 0, but the perturbed render sits next to row 2 (same family).
    perturbed = embeddings[[2]] + torch.randn(1, 32, generator=g) * 0.01

    strict = pairing_scores(embeddings, [0], perturbed)
    assert strict["recall"][1] == 0.0  # not the *anchor* itself

    family = pairing_scores(embeddings, [0], perturbed, base_groups=labels)
    assert family["recall"][1] == 1.0  # but same family is retrieved


def test_nearest_neighbours_lie_in_query_cluster() -> None:
    embeddings, labels = _clustered()
    neighbours = nearest_neighbours(embeddings, labels, [0, 9, 20], k=3)
    for qi, row in zip([0, 9, 20], neighbours):
        for idx, label, _score in row:
            assert label == labels[qi]
            assert idx != qi


def test_cross_set_purity_transfers_label_structure() -> None:
    embeddings, labels = _clustered()
    g = torch.Generator().manual_seed(3)
    queries = embeddings + torch.randn(embeddings.shape, generator=g) * 0.01
    result = cross_set_purity(queries, labels, embeddings, labels, k=3)
    assert result["overall"] > 0.95


def test_similarity_vector_has_one_entry_per_pair() -> None:
    assert similarity_vector(torch.randn(6, 4)).numel() == 15


def test_signal_attribution_flags_the_driving_descriptor() -> None:
    embeddings, _ = _clustered(n_labels=5, per_label=8, dim=32)
    g = torch.Generator().manual_seed(5)
    shape = embeddings.clone()  # identical structure -> should explain everything
    proportion = torch.randn(embeddings.shape[0], 8, generator=g)

    att = signal_attribution(embeddings, {"shape": shape, "proportion": proportion})
    assert att["shape"]["pearson"] > 0.99
    assert abs(att["proportion"]["pearson"]) < 0.5
    assert att["joint"]["r2"] > 0.95
    assert att["joint"]["beta_shape"] > att["joint"]["beta_proportion"]


def test_membership_ranking_recovers_own_font() -> None:
    dim = 16
    centres = torch.eye(3, dim) * 5.0
    g = torch.Generator().manual_seed(6)
    font_embeddings = centres + torch.randn(3, dim, generator=g) * 0.01

    glyphs, own = [], []
    for font in range(3):
        for _ in range(4):
            glyphs.append(centres[font] + torch.randn(dim, generator=g) * 0.05)
            own.append(font)
    result = membership_ranking(torch.stack(glyphs), font_embeddings, own)
    assert result["recall"][1] == 1.0
    assert result["mrr"] == 1.0
    assert result["auc"] == 1.0

    # If the glyphs are assigned to the wrong fonts, the score must collapse.
    scrambled = [own[i] for i in torch.randperm(len(own), generator=g).tolist()]
    assert (
        membership_ranking(torch.stack(glyphs), font_embeddings, scrambled)["auc"] < 0.8
    )
