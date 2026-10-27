from __future__ import annotations

import pytest
import torch
from transformers import PreTrainedTokenizerBase

import sentence_transformers.sentence_transformer.losses.cached_gist_embed as cge
import sentence_transformers.sentence_transformer.losses.gist_embed as ge
from sentence_transformers.sentence_transformer.losses import CachedGISTEmbedLoss, GISTEmbedLoss


class _Tokenizer(PreTrainedTokenizerBase):
    def get_vocab(self) -> dict[str, int]:
        return {"token": 0}


class _EmbeddingModel(torch.nn.Module):
    def __init__(self, guide: bool = False) -> None:
        super().__init__()
        self.tokenizer = _Tokenizer()
        self.max_seq_length = 8
        self.guide = guide
        self.weight = torch.nn.Parameter(torch.eye(3))

    def __getitem__(self, index: int) -> torch.nn.Module:
        return torch.nn.Identity()

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inputs = features["input_ids"]
        embeddings = torch.ones_like(inputs) if self.guide else inputs @ self.weight
        return {"sentence_embedding": embeddings}


@pytest.mark.parametrize("loss_class", [GISTEmbedLoss, CachedGISTEmbedLoss])
@pytest.mark.parametrize(
    "contrast_anchors,contrast_positives,batch_size,rank,with_negative,margin_strategy",
    [
        pytest.param(True, False, 3, 0, False, "absolute", id="anchors-only"),
        pytest.param(False, True, 3, 0, False, "absolute", id="positives-only"),
        pytest.param(True, True, 1, 0, False, "absolute", id="single-pair"),
        pytest.param(True, True, 3, 1, True, "absolute", id="rank1-with-negative"),
        pytest.param(True, True, 3, 1, True, "relative", id="relative-rank1"),
        pytest.param(False, False, 3, 0, True, "absolute", id="no-self-terms-control"),
    ],
)
def test_gist_excludes_self_pairs_when_guide_scores_tie(
    monkeypatch,
    loss_class,
    contrast_anchors: bool,
    contrast_positives: bool,
    batch_size: int,
    rank: int,
    with_negative: bool,
    margin_strategy: str,
) -> None:
    def gather(embeddings: torch.Tensor) -> torch.Tensor:
        return torch.cat([embeddings.flip(0), embeddings]) if rank else embeddings

    for module in (ge, cge):
        monkeypatch.setattr(module, "all_gather_with_grad", gather)
        monkeypatch.setattr(module, "get_rank", lambda: rank)

    generator = torch.Generator().manual_seed(42)
    features = [
        {"input_ids": torch.randn(batch_size, 3, generator=generator)} for _ in range(3 if with_negative else 2)
    ]
    model = _EmbeddingModel()
    kwargs = {"mini_batch_size": 2} if loss_class is CachedGISTEmbedLoss else {}
    loss = loss_class(
        model,
        _EmbeddingModel(guide=True),
        temperature=0.5,
        margin=0.0,
        margin_strategy=margin_strategy,
        contrast_anchors=contrast_anchors,
        contrast_positives=contrast_positives,
        gather_across_devices=True,
        **kwargs,
    )
    actual = loss(features, None)
    actual.backward()
    actual_grad = model.weight.grad.clone()
    model.weight.grad = None

    embeddings = [model(feature)["sentence_embedding"] for feature in features]
    anchor, positive = embeddings[:2]
    candidates = gather(positive)
    rows = torch.arange(batch_size)
    offset = rank * batch_size
    cosine = torch.nn.CosineSimilarity(dim=-1)

    def similarities(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return cosine(left[:, None], right[None, :])

    blocks = [similarities(anchor, candidates)]
    if contrast_anchors:
        block = similarities(anchor, anchor)
        block[rows, rows] = -torch.inf
        blocks.append(block)
    if contrast_positives:
        block = similarities(positive, candidates)
        block[rows, offset + rows] = -torch.inf
        blocks.append(block)
    if with_negative:
        blocks.append(similarities(anchor, gather(embeddings[2])))
    expected = torch.nn.functional.cross_entropy(torch.cat(blocks, dim=1) / 0.5, offset + rows)
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, model.weight.grad, atol=1e-6, rtol=1e-5)
