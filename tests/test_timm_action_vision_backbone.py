from __future__ import annotations

import pytest
import torch
from torch import nn

from va_compound.backbones import TimmActionVisionBackbone


class _FakeTimmDino(nn.Module):
    def __init__(self, container: type[list] | type[tuple]) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.container = container
        self.call: dict[str, object] | None = None

    def get_intermediate_layers(self, images: torch.Tensor, **kwargs):
        self.call = {"images": images, **kwargs}
        patches = [images.new_zeros(images.shape[0], 256, 1024) for _ in range(2)]
        prefixes = [images.new_zeros(images.shape[0], 5, 1024) for _ in range(2)]
        return self.container(zip(patches, prefixes))


@pytest.mark.parametrize("container", [tuple, list])
def test_timm_action_vision_uses_cross_version_intermediate_api(container) -> None:
    model = _FakeTimmDino(container)
    backbone = TimmActionVisionBackbone(model)

    outputs = backbone.forward_hierarchical_dense(
        torch.ones(1, 3, 224, 224, dtype=torch.float64)
    )

    assert set(outputs) == {5, 11}
    assert all(tensor.shape == (1, 256, 1024) for tensor in outputs.values())
    assert model.call is not None
    assert model.call["n"] == [11, 23]
    assert model.call["reshape"] is False
    assert model.call["return_prefix_tokens"] is True
    assert model.call["norm"] is True
    assert model.call["images"].dtype == model.anchor.dtype
