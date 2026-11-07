from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.util import cos_sim


class TinyEncoder(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False, dtype=dtype)

    def __getitem__(self, index):
        return self.projection

    def forward(self, features):
        return {"sentence_embedding": self.projection(features["input_ids"])}


def reference_loss(reps, similarity, directions, partition_mode, hardness_mode):
    """Construct each row's candidates separately and use stable cross entropy."""
    q, p, n = reps
    matrices = {
        "query_to_doc": similarity(q, torch.cat([p, n])).double(),
        "query_to_query": similarity(q, q).double(),
        "doc_to_query": similarity(q, p).T.double(),
        "doc_to_doc": similarity(torch.cat([p, n]), p).T.double(),
    }
    rows = []
    for i in range(len(q)):
        groups = []
        for direction in directions:
            logits = []
            for j, score in enumerate(matrices[direction][i]):
                if direction in ("query_to_query", "doc_to_doc") and j % len(q) == i:
                    continue
                penalty = 0.0
                if direction == "query_to_doc":
                    own = j % len(q) == i
                    hard = own and j >= len(q)
                    penalized = (
                        (hardness_mode == "all_negatives" and j != i)
                        or (hardness_mode == "hard_negatives" and hard)
                        or (hardness_mode == "in_batch_negatives" and not own)
                    )
                    if penalized:
                        penalty = 9.0 * score.detach()
                logits.append(score * 20.0 + penalty)
            groups.append(torch.stack(logits))
        if partition_mode == "joint":
            rows.append(F.cross_entropy(torch.cat(groups)[None], torch.tensor([i])))
        else:
            # Both directions contain the positive at column i.
            rows.append(torch.stack([F.cross_entropy(group[None], torch.tensor([i])) for group in groups]).mean())
    return torch.stack(rows).mean()


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    "directions,partition_mode,hardness_mode",
    [
        (("query_to_doc",), "joint", None),
        (("query_to_doc",), "joint", "hard_negatives"),
        (("query_to_doc",), "joint", "in_batch_negatives"),
        (("query_to_doc",), "joint", "all_negatives"),
        (("query_to_doc", "doc_to_query"), "per_direction", None),
        (("query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"), "joint", "all_negatives"),
    ],
)
def test_mnrl_loss_arithmetic_precision(cached, dtype, directions, partition_mode, hardness_mode):
    torch.manual_seed(17)
    model = TinyEncoder()
    expected_model = deepcopy(model)
    features = [{"input_ids": torch.randn(5, 4, dtype=torch.float64)} for _ in range(3)]

    def similarity(a, b):
        return cos_sim(a, b).to(dtype)

    cls = CachedMultipleNegativesRankingLoss if cached else MultipleNegativesRankingLoss
    loss_fn = cls(
        model,
        similarity_fct=similarity,
        directions=directions,
        partition_mode=partition_mode,
        hardness_mode=hardness_mode,
        hardness_strength=9.0 if hardness_mode else 0.0,
        **({"mini_batch_size": 2} if cached else {}),
    )
    loss = loss_fn(features, None)
    expected = reference_loss(
        [expected_model(f)["sentence_embedding"] for f in features],
        similarity,
        directions,
        partition_mode,
        hardness_mode,
    )
    assert loss.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    torch.testing.assert_close(loss.double(), expected, rtol=2e-6, atol=2e-6)
    (loss / 4).backward()
    (expected / 4).backward()
    tolerance = 0.01 if dtype == torch.bfloat16 else 0.001 if dtype == torch.float16 else 2e-6
    torch.testing.assert_close(
        model.projection.weight.grad, expected_model.projection.weight.grad, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("cached", [False, True])
def test_mnrl_cpu_autocast_training(cached):
    torch.manual_seed(17)
    model = TinyEncoder(dtype=torch.float32)
    expected_model = deepcopy(model)
    features = [{"input_ids": torch.randn(5, 4)} for _ in range(3)]
    cls = CachedMultipleNegativesRankingLoss if cached else MultipleNegativesRankingLoss
    loss_fn = cls(model, **({"mini_batch_size": 2} if cached else {}))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = loss_fn(features, None)
        reps = [expected_model(f)["sentence_embedding"] for f in features]
        logits = cos_sim(reps[0], torch.cat(reps[1:])).float() * 20
        expected = F.cross_entropy(logits, torch.arange(5))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    expected.backward()
    relative_error = (model.projection.weight.grad - expected_model.projection.weight.grad).norm()
    relative_error /= expected_model.projection.weight.grad.norm()
    assert relative_error < 0.01
