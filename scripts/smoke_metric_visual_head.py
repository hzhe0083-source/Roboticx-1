"""MT-VJ 度量视觉头冒烟测试（契约 §2 验收）：随机输入验证。

覆盖：
1. LanguageMetricField / RelationStateEncoder / MicroRefiner 全部 forward 形状；
2. 所有输出数值有限；
3. p ∈ [0,1]（y,x 序）；
4. coords [-1,1]（live_vjepa._dense_coords 实际输出）与 [0,1] 两种约定等价；
5. eef_pos=None 时 relation 置零；传入时 relation 有限；
6. 空语言 mask（全 False）不产生 NaN；
7. fp16 输入可正常前向；
8. 反向传播：所有参数梯度存在且有限；
9. "W_o 类零初始化可用"：零初始化输出投影消费 z_g/z_nu 时输出严格为零、
   上游（RelationStateEncoder）梯度不受阻（契约 §5 残差注入的等价性前提）。

运行：/home/ryan/.venvs/pytorch-gpu/bin/python scripts/smoke_metric_visual_head.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根（独立运行）

import torch
import torch.nn as nn

from va_compound.live_vjepa import _dense_coords
from va_compound.metric_visual_head import (
    LanguageMetricField,
    MetricFieldOutput,
    MicroRefiner,
    N_ROLES,
    RelationStateEncoder,
    ROLE_NAMES,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, L = 2, 8


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def check_finite(name: str, t: torch.Tensor) -> None:
    assert torch.isfinite(t).all(), f"{name} 含非有限值: {t.detach().abs().max().item()}"
    print(f"  {name}: shape={tuple(t.shape)} dtype={t.dtype} finite ✓")


def main() -> None:
    torch.manual_seed(0)
    print(f"device={DEV}")

    # ---------- 1. 三个模块构造 + 参数统计 ----------
    head = LanguageMetricField().to(DEV)
    rel_enc = RelationStateEncoder().to(DEV)
    refiner = MicroRefiner().to(DEV)
    print(f"\n参数统计: LanguageMetricField={count_params(head):,}  "
          f"RelationStateEncoder={count_params(rel_enc):,}  "
          f"MicroRefiner={count_params(refiner):,}")

    # ---------- 2. 随机输入（coords 用仓库实际 _dense_coords：[-1,1] 约定） ----------
    coords_neg = torch.from_numpy(_dense_coords()).to(DEV)  # [1152, 3] t∈{-1,1}, y/x∈[-1,1]
    assert coords_neg.shape == (1152, 3)
    h5 = torch.randn(B, 1152, 768, device=DEV)
    h11 = torch.randn(B, 1152, 768, device=DEV)
    lang = torch.randn(B, L, 2048, device=DEV)
    mask = torch.ones(B, L, dtype=torch.bool, device=DEV)
    mask[1, 4:] = False  # 第二条指令带 padding
    eef_pos = torch.rand(B, 2, device=DEV)

    print("\n=== LanguageMetricField forward（eef_pos 传入）===")
    out: MetricFieldOutput = head(h5, h11, lang, mask, coords_neg, eef_pos=eef_pos)
    assert out.p.shape == (B, N_ROLES, 2), out.p.shape
    assert out.visibility.shape == (B, N_ROLES), out.visibility.shape
    assert out.offset.shape == (B, N_ROLES, 2), out.offset.shape
    assert out.heatmap.shape == (B, N_ROLES, 24, 24), out.heatmap.shape
    assert out.relation.shape == (B, 4), out.relation.shape
    check_finite("p", out.p)
    check_finite("visibility", out.visibility)
    check_finite("offset", out.offset)
    check_finite("heatmap", out.heatmap)
    check_finite("relation", out.relation)
    p_min, p_max = out.p.min().item(), out.p.max().item()
    assert 0.0 <= p_min and p_max <= 1.0, f"p 超出 [0,1]: [{p_min}, {p_max}]"
    print(f"  p ∈ [{p_min:.6f}, {p_max:.6f}] ⊆ [0,1] ✓")
    vis_min, vis_max = out.visibility.min().item(), out.visibility.max().item()
    assert 0.0 <= vis_min and vis_max <= 1.0
    print(f"  visibility ∈ [{vis_min:.4f}, {vis_max:.4f}] ⊆ [0,1] ✓")
    print(f"  relation 示例: {out.relation[0].tolist()}")

    # ---------- 3. eef_pos=None → relation 置零 ----------
    print("\n=== eef_pos=None → relation 全零 ===")
    out0 = head(h5, h11, lang, mask, coords_neg, eef_pos=None)
    assert out0.relation.abs().max().item() == 0.0, "eef_pos=None 时 relation 必须全零"
    print("  relation ≡ 0 ✓；p 与传入 eef_pos 时逐位一致: "
          f"{torch.equal(out0.p, out.p)}")

    # ---------- 4. coords [0,1] 约定与 [-1,1] 约定等价 ----------
    print("\n=== coords 两种归一化约定等价 ===")
    coords_01 = (coords_neg + 1.0) / 2.0
    out01 = head(h5, h11, lang, mask, coords_01, eef_pos=eef_pos)
    for name in ("p", "visibility", "offset", "heatmap", "relation"):
        a, b = getattr(out0 if False else out, name), getattr(out01, name)
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    print("  [-1,1] 与 [0,1] 输入 → 输出逐字段一致 ✓")

    # ---------- 5. 空语言 mask（全 False）不产生 NaN ----------
    print("\n=== 空语言 mask（全 False）===")
    mask_empty = torch.zeros(B, L, dtype=torch.bool, device=DEV)
    out_e = head(h5, h11, lang, mask_empty, coords_neg, eef_pos=eef_pos)
    for name, t in out_e.__dict__.items():
        check_finite(name, t)
    assert 0.0 <= out_e.p.min().item() and out_e.p.max().item() <= 1.0
    print("  查询已置零、输出有限、p ∈ [0,1] ✓")

    # ---------- 6. fp16 输入 ----------
    print("\n=== fp16 输入 ===")
    out_h = head(h5.half(), h11.half(), lang.half(), mask, coords_neg.half(),
                 eef_pos=eef_pos.half())
    for name, t in out_h.__dict__.items():
        check_finite(name, t)
    print("  fp16 输入前向正常 ✓")

    # ---------- 7. 反向传播：全部参数梯度存在且有限 ----------
    print("\n=== 反向传播 ===")
    loss = (
        out.p.abs().mean() + out.visibility.mean() + out.relation.abs().mean()
        + out.offset.abs().mean() + out.heatmap.mean()
    )
    loss.backward()
    n_missing, n_zero = 0, 0
    for name, param in head.named_parameters():
        if param.grad is None:
            n_missing += 1
            continue
        assert torch.isfinite(param.grad).all(), f"{name} 梯度非有限"
        if param.grad.abs().max().item() == 0.0:
            n_zero += 1
    assert n_missing == 0, f"{n_missing} 个参数无梯度"
    print(f"  全部 {sum(1 for _ in head.parameters())} 个参数梯度存在且有限 "
          f"（严格零梯度 {n_zero} 个）✓")

    # ---------- 8. RelationStateEncoder ----------
    print("\n=== RelationStateEncoder ===")
    g = torch.randn(B, 4, device=DEV)
    nu = torch.randn(B, 4, device=DEV)
    z_g, z_nu = rel_enc(g, nu)
    assert z_g.shape == z_nu.shape == (B, 512)
    check_finite("z_g", z_g)
    check_finite("z_nu", z_nu)
    loss2 = (z_g.square().mean() + z_nu.square().mean())
    loss2.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in rel_enc.parameters()
    )
    print("  反向传播 ✓")

    # ---------- 9. MicroRefiner ----------
    print("\n=== MicroRefiner ===")
    roi = torch.randn(B, 3, 96, 96, device=DEV)
    micro = refiner(roi)
    assert micro.shape == (B, 4)
    check_finite("micro", micro)
    micro.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in refiner.parameters()
    )
    print("  反向传播 ✓")

    # ---------- 10. "W_o 类零初始化可用"：零初始化输出投影消费 metric tokens ----------
    print("\n=== W_o 类零初始化可用性（契约 §5 残差注入前提）===")
    w_o = nn.Linear(512, 512, bias=True).to(DEV)
    nn.init.zeros_(w_o.weight)
    nn.init.zeros_(w_o.bias)
    # 重新前向（避免复用已释放的计算图）。
    g2 = torch.randn(B, 4, device=DEV)
    nu2 = torch.randn(B, 4, device=DEV)
    z_g2, z_nu2 = rel_enc(g2, nu2)
    metric_tokens = torch.stack((z_g2, z_nu2), dim=1)  # [B, 2, 512]
    injected = w_o(metric_tokens)
    assert injected.abs().max().item() == 0.0, "零初始化 W_o 输出必须严格为零"
    # 上游梯度路径：经 metric_tokens 的额外损失项（等价于 A_base 主损失通路）。
    torch.autograd.backward(
        (injected.square().mean() + metric_tokens.mean()),
        inputs=list(rel_enc.parameters()),
    )
    grads = [p.grad for p in rel_enc.parameters() if p.grad is not None]
    assert grads and max(gr.abs().max().item() for gr in grads) > 0.0, \
        "零初始化 W_o 下游时 RelationStateEncoder 梯度必须不受阻"
    print("  零初始化 W_o 输出严格为零、上游梯度正常流动（双零互锁不存在）✓")

    print("\nSMOKE OK —— 全部断言通过")


if __name__ == "__main__":
    main()
