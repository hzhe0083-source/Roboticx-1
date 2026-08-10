"""MT-VJ §1 冒烟：VJEPA21Backbone.forward_hierarchical_dense 形状/有限性验证。

随机输入 [2, 4, 3, 384, 384]（fp16，与 prepare_pnpw_features 提取同构），
一次前向 → {5: [2, 1152, 768], 11: [2, 1152, 768]}（未池化全 patch，t→y→x 序）。
权重：本地官方 vjepa2_1_vitb_dist_vitG_384.pt（local_files_only=True）。

额外验证（契约 §8 一致性的冒烟版）：
- out_layers=(11,) 的输出应与 _encode 逐位一致（官方 forward 在 block 11
  后收集 self.norm(x)，与缺省路径末尾 norm 是同一调用；_pool("dense") 只
  是等形状 reshape）。
- 前向前后缺省路径输出逐位不变（encode_multi 的临时 out_layers 改写已恢复）。

用法：PYTHONPATH=. /home/ryan/.venvs/pytorch-gpu/bin/python \
      scripts/smoke_forward_hierarchical_dense.py
"""
from __future__ import annotations

import torch

from va_compound.backbones import VJEPA21Backbone

B, W = 2, 4


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype="float16", local_files_only=True
    )
    backbone.eval()
    print(f"backbone: {device} fp16, blocks={len(backbone.model.blocks)}, "
          f"patch_grid={backbone.patch_grid()}")

    video = torch.randn(B, W, 3, 384, 384, device=device, dtype=torch.float16)
    with torch.no_grad():
        default_before = backbone._encode(video)  # 缺省路径基线
        out = backbone.forward_hierarchical_dense(video, out_layers=(5, 11))
        default_after = backbone._encode(video)  # 缺省路径基线（调用后）

    print(f"video: {tuple(video.shape)}")
    assert set(out.keys()) == {5, 11}, out.keys()
    for layer in (5, 11):
        t = out[layer]
        print(f"layer {layer}: shape={tuple(t.shape)} dtype={t.dtype} "
              f"finite={bool(torch.isfinite(t).all())} "
              f"abs_max={float(t.abs().max())} abs_mean={float(t.abs().mean())}")
        assert tuple(t.shape) == (B, 1152, 768), tuple(t.shape)
        assert bool(torch.isfinite(t).all())
        assert not bool(torch.isnan(t).any()) and not bool(torch.isinf(t).any())

    # 契约 §8 一致性：_encode ≡ out_layers=(11,) 池化前（逐位）。
    assert torch.equal(default_before, default_after), "缺省路径被调用改变"
    assert torch.equal(default_before, out[11]), "_encode 与 layer 11 输出不一致"
    print("consistency: _encode == out[11] bitwise OK, default path unchanged OK")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
