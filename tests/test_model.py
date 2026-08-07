import unittest

import torch
from torch import nn

from va_compound.backbones import VJEPA21Backbone
from va_compound.model import VACompoundConfig, VACompoundPolicy, VACouplingLayer


def tiny_config(mode: str = "bidir_va") -> VACompoundConfig:
    return VACompoundConfig(
        language_dim=24,
        vision_dim=20,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        action_horizon=5,
        action_dim=6,
        proprio_dim=9,
        mode=mode,
    )


class VACompoundTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def inputs(self, config: VACompoundConfig):
        return (
            torch.randn(2, 11, config.vision_dim),
            torch.randn(2, config.proprio_dim),
            torch.randn(2, config.action_dim),
            torch.randn(2, 7, config.language_dim),
            torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool),
        )

    def flow_inputs(self, config: VACompoundConfig):
        return (
            torch.randn(2, config.action_horizon, config.action_dim),
            torch.rand(2),
        )

    def test_forward_backward_and_shape(self) -> None:
        config = tiny_config()
        model = VACompoundPolicy(config)
        vision, proprio, previous, language, mask = self.inputs(config)
        noisy_actions, flow_time = self.flow_inputs(config)
        predicted = model(
            vision,
            proprio,
            previous,
            noisy_actions,
            flow_time,
            language_hidden=language,
            language_mask=mask,
        )
        self.assertEqual(predicted.shape, (2, config.action_horizon, config.action_dim))
        loss = predicted.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(model.action_queries.grad).all())

    def test_cached_and_uncached_language_match(self) -> None:
        config = tiny_config()
        model = VACompoundPolicy(config).eval()
        vision, proprio, previous, language, mask = self.inputs(config)
        noisy_actions, flow_time = self.flow_inputs(config)
        cache = model.build_language_cache(language, mask)
        direct = model(
            vision,
            proprio,
            previous,
            noisy_actions,
            flow_time,
            language_hidden=language,
            language_mask=mask,
        )
        cached = model(
            vision,
            proprio,
            previous,
            noisy_actions,
            flow_time,
            language_cache=cache,
        )
        torch.testing.assert_close(direct, cached)

    def test_detached_cache_supports_repeated_backward(self) -> None:
        config = tiny_config()
        model = VACompoundPolicy(config)
        vision, proprio, previous, language, mask = self.inputs(config)
        noisy_actions, flow_time = self.flow_inputs(config)
        cache = model.build_language_cache(language, mask, detach=True)
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            model(
                vision,
                proprio,
                previous,
                noisy_actions,
                flow_time,
                language_cache=cache,
            ).square().mean().backward()
        self.assertTrue(torch.isfinite(model.action_queries.grad).all())

    def test_cache_to_preserves_mask_and_converts_projected_tensors(self) -> None:
        config = tiny_config()
        model = VACompoundPolicy(config)
        _, _, _, language, mask = self.inputs(config)
        cache = model.build_language_cache(language, mask, detach=True).to(dtype=torch.float64)
        self.assertEqual(cache.layers[0].key.dtype, torch.float64)
        self.assertEqual(cache.attention_mask.dtype, torch.bool)

    def test_language_mask_shape_is_validated(self) -> None:
        config = tiny_config()
        model = VACompoundPolicy(config)
        language = torch.randn(2, 7, config.language_dim)
        with self.assertRaisesRegex(ValueError, "language_mask"):
            model.build_language_cache(language, torch.ones(2, 6, dtype=torch.bool))

    def test_baseline_and_proposal_have_equal_parameter_counts(self) -> None:
        proposal = VACompoundPolicy(tiny_config("bidir_va"))
        baseline = VACompoundPolicy(tiny_config("uni_a"))
        proposal_count = sum(parameter.numel() for parameter in proposal.parameters())
        baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
        self.assertEqual(proposal_count, baseline_count)

    def test_uni_a_removes_language_and_action_updates_from_visual_queries(self) -> None:
        layer = VACouplingLayer(
            hidden_dim=32,
            language_dim=24,
            num_heads=4,
            dropout=0.0,
            mode="uni_a",
        ).eval()
        visual = torch.randn(2, 6, 32)
        action = torch.randn(2, 5, 32)
        language_a = torch.randn(2, 4, 24)
        language_b = torch.randn(2, 4, 24)
        memory_a = torch.randn(2, 6, 32)
        memory_b = torch.randn(2, 6, 32)
        mask = torch.ones(2, 4, dtype=torch.bool)
        visual_a, action_a, _ta = layer(
            visual,
            action,
            layer.project_language(language_a),
            mask,
            visual_memory=memory_a,
        )
        visual_b, action_b, _tb = layer(
            visual,
            action,
            layer.project_language(language_b),
            mask,
            visual_memory=memory_b,
        )
        torch.testing.assert_close(visual_a, visual_b)
        self.assertFalse(torch.allclose(action_a, action_b))

    def test_bidir_va_allows_language_to_update_visual_tokens(self) -> None:
        layer = VACouplingLayer(
            hidden_dim=32,
            language_dim=24,
            num_heads=4,
            dropout=0.0,
            mode="bidir_va",
        ).eval()
        visual = torch.randn(2, 6, 32)
        action = torch.randn(2, 5, 32)
        mask = torch.ones(2, 4, dtype=torch.bool)
        visual_a, _, _ta = layer(visual, action, layer.project_language(torch.randn(2, 4, 24)), mask)
        visual_b, _, _tb = layer(visual, action, layer.project_language(torch.randn(2, 4, 24)), mask)
        self.assertFalse(torch.allclose(visual_a, visual_b))

    def test_bidir_va_allows_previous_visual_to_update_current_visual(self) -> None:
        layer = VACouplingLayer(
            hidden_dim=32,
            language_dim=24,
            num_heads=4,
            dropout=0.0,
            mode="bidir_va",
        ).eval()
        visual = torch.randn(2, 6, 32)
        action = torch.randn(2, 5, 32)
        language = layer.project_language(torch.randn(2, 4, 24))
        mask = torch.ones(2, 4, dtype=torch.bool)
        visual_a, _, _ta = layer(
            visual,
            action,
            language,
            mask,
            visual_memory=torch.randn(2, 6, 32),
        )
        visual_b, _, _tb = layer(
            visual,
            action,
            language,
            mask,
            visual_memory=torch.randn(2, 6, 32),
        )
        self.assertFalse(torch.allclose(visual_a, visual_b))

    def test_previous_goal_visual_changes_next_action(self) -> None:
        config = tiny_config()
        model = VACompoundPolicy(config).eval()
        vision, proprio, previous, language, mask = self.inputs(config)
        noisy_actions, flow_time = self.flow_inputs(config)
        cache = model.build_language_cache(language, mask)
        _, memory = model(
            vision,
            proprio,
            previous,
            noisy_actions,
            flow_time,
            language_cache=cache,
            return_visual_memory=True,
        )
        next_vision = torch.randn_like(vision)
        without_memory = model(
            next_vision,
            proprio,
            previous,
            noisy_actions,
            flow_time,
            language_cache=cache,
        )
        with_memory, next_memory = model(
            next_vision,
            proprio,
            previous,
            noisy_actions,
            flow_time,
            language_cache=cache,
            visual_memory=memory,
            return_visual_memory=True,
        )
        self.assertEqual(len(memory.layers), config.num_layers)
        self.assertEqual(next_memory.layers[-1].shape, (2, 11, config.hidden_dim))
        self.assertFalse(torch.allclose(without_memory, with_memory))

    def test_temporal_action_loss_trains_memory_projections(self) -> None:
        config = tiny_config()
        model = VACompoundPolicy(config)
        vision, proprio, previous, language, mask = self.inputs(config)
        noisy_actions, flow_time = self.flow_inputs(config)
        cache = model.build_language_cache(language, mask)
        _, memory = model(
            vision,
            proprio,
            previous,
            noisy_actions,
            flow_time,
            language_cache=cache,
            return_visual_memory=True,
        )
        predicted, _ = model(
            torch.randn_like(vision),
            proprio,
            previous,
            noisy_actions,
            flow_time,
            language_cache=cache,
            visual_memory=memory,
            return_visual_memory=True,
        )
        predicted.square().mean().backward()
        gradient = model.layers[0].k_m.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_flow_sampler_reuses_one_encoded_condition(self) -> None:
        config = tiny_config()
        model = VACompoundPolicy(config).eval()
        vision, proprio, previous, language, mask = self.inputs(config)
        condition = model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
        )
        noise, _ = self.flow_inputs(config)
        sampled = model.sample_actions(condition, steps=4, noise=noise)
        self.assertEqual(sampled.shape, noise.shape)
        self.assertTrue(torch.isfinite(sampled).all())

    def test_flow_matching_loss_is_zero_for_exact_velocity(self) -> None:
        velocity = torch.randn(2, 5, 6)
        self.assertEqual(VACompoundPolicy.flow_matching_loss(velocity, velocity).item(), 0.0)

    def test_unfreeze_zero_keeps_vjepa_frozen(self) -> None:
        class FakeVJEPA(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
                self.norms_block = nn.ModuleList([nn.LayerNorm(2)])

        backbone = VJEPA21Backbone(FakeVJEPA())
        backbone.unfreeze_last(0)
        self.assertTrue(all(not parameter.requires_grad for parameter in backbone.parameters()))

    def test_unfreeze_last_only_opens_requested_block_and_norm(self) -> None:
        class FakeVJEPA(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
                self.norms_block = nn.ModuleList([nn.LayerNorm(2)])

        backbone = VJEPA21Backbone(FakeVJEPA())
        backbone.unfreeze_last(1)
        self.assertTrue(all(not p.requires_grad for p in backbone.model.blocks[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in backbone.model.blocks[1].parameters()))
        self.assertTrue(all(p.requires_grad for p in backbone.model.norms_block[-1].parameters()))


if __name__ == "__main__":
    unittest.main()


class SMCAttnTests(unittest.TestCase):
    """SMC-Attn：源测度校正注意力（2026-08-05 原创设计）。

    核心性质：同一来源的 token 被复制 r 倍后，SMC 下动作输出保持不变
    （-log N_s 精确抵消分母增加）；flat 下输出必然改变（token 数量买票）。
    """

    def setUp(self) -> None:
        torch.manual_seed(11)

    def _layer(self, variant: str) -> VACouplingLayer:
        cfg = VACompoundConfig(
            language_dim=24,
            vision_dim=20,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            action_horizon=2,
            action_dim=6,
            proprio_dim=9,
            mode="uni_a",
            attention_variant=variant,
        )
        return VACouplingLayer(
            language_dim=cfg.language_dim,
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=0.0,
            mode=cfg.mode,
            qk_norm=False,
            attention_variant=cfg.attention_variant,
        )

    def _run(self, layer: VACouplingLayer, visual, action, language, mask):
        layer.eval()
        with torch.no_grad():
            return layer.forward(
                visual,
                action,
                language,
                mask,
                visual_memory=None,
            )  # (visual 更新, action 更新)

    def test_smc_token_refinement_invariance(self) -> None:
        """复制 V/M token 2x：SMC 动作输出不变（≤1e-4），flat 变化（>1e-2）。"""
        from va_compound.model import LayerLanguageCache

        B, NV, NA, NL, H = 2, 4, 2, 3, 32
        v = torch.randn(B, NV, H)  # visual 已是 hidden_dim 空间
        a = torch.randn(B, NA, H)
        # language cache = 投影后分头的 key/value（policy 侧 build_language_cache 产物）
        n_heads, head_dim = 4, H // 4
        lang = LayerLanguageCache(
            torch.randn(B, n_heads, NL, head_dim),
            torch.randn(B, n_heads, NL, head_dim),
        )
        mask = torch.ones(B, NL, dtype=torch.bool)

        smc = self._layer("smc")
        flat = self._layer("flat")

        out_smc = self._run(smc, v, a, lang, mask)[1]  # action 更新
        out_flat = self._run(flat, v, a, lang, mask)[1]

        # 复制 V token 2 倍（n_visual 4 → 8）
        v2 = torch.cat([v, v], dim=1)
        out_smc2 = self._run(smc, v2, a, lang, mask)[1]
        out_flat2 = self._run(flat, v2, a, lang, mask)[1]

        d_smc = (out_smc - out_smc2).abs().max().item()
        d_flat = (out_flat - out_flat2).abs().max().item()
        self.assertLess(d_smc, 1e-4, f"SMC 应保持复制不变性，实际位移 {d_smc:.6f}")
        self.assertGreater(d_flat, 1e-2, f"flat 应随 token 数量改变，实际位移 {d_flat:.6f}")

    def test_smc_invalid_variant_rejected(self) -> None:
        with self.assertRaises(ValueError):
            VACompoundConfig(
                language_dim=24,
                vision_dim=20,
                hidden_dim=32,
                num_layers=1,
                num_heads=4,
                action_horizon=2,
                action_dim=6,
                proprio_dim=9,
                attention_variant="bogus",
            )

    def test_action_query_cond_zero_init_equals_static(self) -> None:
        """开启 action_query_cond 且 zero-init 未训练时，输出必须与静态 query 基线一致。"""
        base_cfg = tiny_config()
        cond_cfg = VACompoundConfig(**{**base_cfg.__dict__, "action_query_cond": True})
        base = VACompoundPolicy(base_cfg).eval()
        cond = VACompoundPolicy(cond_cfg).eval()
        # 把 cond 的非 lang_to_query 权重复制为 base（lang_to_query 保持 zero-init）
        cond.load_state_dict(
            {k: v for k, v in base.state_dict().items()}, strict=False
        )
        torch.manual_seed(3)
        B, T, TD, L = 2, 4, 20, 7
        vision = torch.randn(B, T, TD)
        proprio = torch.randn(B, 9)
        previous = torch.randn(B, 6)
        lang_h = torch.randn(B, L, base_cfg.language_dim)
        lang_m = torch.ones(B, L, dtype=torch.bool)
        with torch.inference_mode():
            c_base = base.encode_condition(
                vision, proprio, previous,
                language_hidden=lang_h, language_mask=lang_m,
            )
            c_cond = cond.encode_condition(
                vision, proprio, previous,
                language_hidden=lang_h, language_mask=lang_m,
            )
        torch.testing.assert_close(c_cond, c_base, atol=1e-6, rtol=1e-6)
        # 训练一步后（非零偏移）输出必须改变
        with torch.no_grad():
            for p in cond.lang_to_query.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        with torch.inference_mode():
            c_cond2 = cond.encode_condition(
                vision, proprio, previous,
                language_hidden=lang_h, language_mask=lang_m,
            )
        self.assertFalse(torch.allclose(c_cond2, c_base, atol=1e-4))


class MemorySplitTests(unittest.TestCase):
    """Evidence/Task memory split (2026-08-07 审阅落地②) contract tests."""

    def _policy(self, memory_split: bool):
        cfg = VACompoundConfig(
            language_dim=12,
            vision_dim=10,
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            action_horizon=3,
            action_dim=4,
            proprio_dim=5,
            memory_split=memory_split,
            evidence_tokens=6,
            task_tokens=4,
        )
        return VACompoundPolicy(cfg).eval()

    def test_evidence_is_kv_only_and_preserved_across_steps(self) -> None:
        """Evidence cannot be written by action/language: it enters the layer
        only as a K/V source, so its value is unchanged by attention."""
        m = self._policy(True)
        B = 2
        vis = torch.randn(B, 8, 10)
        prop = torch.randn(B, 5)
        prev = torch.randn(B, 4)
        lang = torch.randn(B, 4, 12)
        mask = torch.ones(B, 4, dtype=torch.bool)
        cond, mem = m.encode_condition(
            vis, prop, prev, language_hidden=lang, language_mask=mask,
            return_visual_memory=True,
        )
        evidence_after_step = mem.evidence.detach().clone()
        # Next step with a completely different action stream: evidence
        # updates only from (vision, state, prev evidence) -> repeat the same
        # vision/state -> same evidence gate output (deterministic).
        cond2, mem2 = m.encode_condition(
            vis, prop, prev, language_hidden=lang, language_mask=mask,
            visual_memory=mem, return_visual_memory=True,
        )
        self.assertEqual(mem.evidence.shape, mem2.evidence.shape)
        self.assertEqual(len(mem.layers), 0)  # no per-layer snapshots in split mode

    def test_first_step_full_overwrite_then_gated(self) -> None:
        """First step (no prior evidence) must fully overwrite the init slots;
        subsequent steps apply the learned gate."""
        m = self._policy(True)
        B = 1
        vis = torch.randn(B, 8, 10)
        prop = torch.randn(B, 5)
        prev = torch.randn(B, 4)
        lang = torch.randn(B, 4, 12)
        mask = torch.ones(B, 4, dtype=torch.bool)
        # First step: evidence must not equal the (untrained) init parameter
        # when the vision content differs from it.
        cond, mem = m.encode_condition(
            vis, prop, prev, language_hidden=lang, language_mask=mask,
            return_visual_memory=True,
        )
        init = m.evidence_init.detach()
        self.assertFalse(torch.allclose(mem.evidence, init.expand(B, -1, -1), atol=1e-3))

    def test_task_workspace_updates_gated_and_conditions_action(self) -> None:
        """Task workspace changes with language and feeds the action
        condition (C_t = LN(A + P_T T))."""
        m = self._policy(True)
        B = 1
        vis = torch.randn(B, 8, 10)
        prop = torch.randn(B, 5)
        prev = torch.randn(B, 4)
        mask = torch.ones(B, 4, dtype=torch.bool)
        lang_a = torch.randn(B, 4, 12)
        lang_b = torch.randn(B, 4, 12)
        cond_a, mem_a = m.encode_condition(
            vis, prop, prev, language_hidden=lang_a, language_mask=mask,
            return_visual_memory=True,
        )
        cond_b = m.encode_condition(
            vis, prop, prev, language_hidden=lang_b, language_mask=mask,
        )
        self.assertFalse(torch.allclose(cond_a, cond_b, atol=1e-4))
        self.assertEqual(mem_a.task.shape, (B, 4, 16))

    def test_legacy_path_identical_shapes(self) -> None:
        """memory_split=False keeps the old per-layer memory contract."""
        m = self._policy(False)
        B = 1
        vis = torch.randn(B, 8, 10)
        prop = torch.randn(B, 5)
        prev = torch.randn(B, 4)
        lang = torch.randn(B, 4, 12)
        mask = torch.ones(B, 4, dtype=torch.bool)
        cond, mem = m.encode_condition(
            vis, prop, prev, language_hidden=lang, language_mask=mask,
            return_visual_memory=True,
        )
        self.assertEqual(len(mem.layers), 2)
        self.assertIsNone(mem.evidence)
        self.assertIsNone(mem.task)


class SequentialAndAdaLNTests(unittest.TestCase):
    """顺序式 A→V→A + flow AdaLN 条件（2026-08-07 审阅落地④）contract tests."""

    def _policy(self, **kw):
        cfg = VACompoundConfig(
            language_dim=12,
            vision_dim=10,
            hidden_dim=16,
            num_layers=4,
            num_heads=4,
            action_horizon=3,
            action_dim=4,
            proprio_dim=5,
            **kw,
        )
        return VACompoundPolicy(cfg).eval()

    def test_sequential_coupling_runs_and_is_deterministic(self) -> None:
        m = self._policy(sequential_coupling=2, memory_split=True,
                         evidence_tokens=6, task_tokens=4)
        B = 1
        vis = torch.randn(B, 8, 10)
        prop = torch.randn(B, 5)
        prev = torch.randn(B, 4)
        lang = torch.randn(B, 4, 12)
        mask = torch.ones(B, 4, dtype=torch.bool)
        torch.manual_seed(1)
        c1, mem1 = m.encode_condition(vis, prop, prev, language_hidden=lang,
                                      language_mask=mask, return_visual_memory=True)
        torch.manual_seed(1)
        c2, mem2 = m.encode_condition(vis, prop, prev, language_hidden=lang,
                                      language_mask=mask, return_visual_memory=True)
        torch.testing.assert_close(c1, c2)
        torch.testing.assert_close(mem1.task, mem2.task)
        self.assertEqual(mem1.evidence.shape, (B, 6, 16))
        # alternating layers: layers 2 and 4 sequential (1-based)
        self.assertTrue(m.layers[1].sequential)
        self.assertTrue(m.layers[3].sequential)
        self.assertFalse(m.layers[0].sequential)

    def test_adaln_flow_zero_init_condition_independent(self) -> None:
        """AdaLN-Zero 零初始化：训练起点 gate=0 → 输出与条件无关（条件通道
        从零开始学，不破坏起点行为）。"""
        torch.manual_seed(0)
        m_ada = self._policy(flow_cond="adaln")
        noisy = torch.randn(2, 3, 4)
        t = torch.zeros(2)
        cond_a = torch.randn(2, 3, 16)
        cond_b = torch.randn(2, 3, 16)
        v_a = m_ada.flow_velocity(cond_a, noisy, t)
        v_b = m_ada.flow_velocity(cond_b, noisy, t)
        torch.testing.assert_close(v_a, v_b, atol=1e-5, rtol=1e-5)


class FuturePredictContractTests(unittest.TestCase):
    """future_predict 训练链契约：batch 维 ≠ 时间维时不得崩溃/错位。

    2026-08-07 回归：train.py 曾用裸索引 ``action_conditions[t]`` 取时间维，
    对 [B, T, H, hidden] 实际取到 batch 维；B=T=4 时形状巧合匹配掩盖了错位，
    每 epoch 最后一个 3 样本 batch（9927 % 4 = 3）在 step 2482 崩溃
    ("Expected size 4 but got size 3")。修复为 ``[:, t]``，此测试用 B≠T 锁定。
    """

    def _policy(self):
        cfg = VACompoundConfig(
            language_dim=12,
            vision_dim=10,
            hidden_dim=16,
            num_layers=4,
            num_heads=4,
            action_horizon=3,
            action_dim=4,
            proprio_dim=5,
            memory_split=True,
            evidence_tokens=6,
            task_tokens=4,
            sequential_coupling=2,
            flow_cond="adaln",
            future_predict=True,
        )
        return VACompoundPolicy(cfg)

    def _run_future_chain(self, B: int) -> None:
        m = self._policy().train()
        T, H, A, V = 4, 3, 4, 10
        vis = torch.randn(B, T, 8, V)   # [B, T, tokens, vision_dim]
        prop = torch.randn(B, T, 5)
        prev = torch.randn(B, T, A)
        lang = torch.randn(B, 4, 12)
        mask = torch.ones(B, 4, dtype=torch.bool)
        noisy = torch.randn(B, T, H, A)
        ftime = torch.rand(B, T)
        lc = m.build_language_cache(lang, mask)
        mem = None
        conds: list[Tensor] = []
        mems: list[VisualMemory] = []
        velos: list[Tensor] = []
        for t in range(T):
            cond, mem = m.encode_condition(
                vis[:, t], prop[:, t], prev[:, t],
                language_cache=lc, visual_memory=mem, return_visual_memory=True,
            )
            conds.append(cond)
            mems.append(mem)
            velos.append(m.flow_velocity(cond, noisy[:, t], ftime[:, t]))
        action_conditions = torch.stack(conds, dim=1)  # 与 train.py rollout_policy 相同
        flow_loss = torch.stack(velos).mean()
        terms = []
        for t in range(T - 1):
            pred = m.future_predictor(
                action_conditions[:, t], mems[t].evidence, mems[t].task
            )
            tgt = vis[:, t + 1].mean(dim=1)
            self.assertEqual(pred.shape, (B, V), f"B={B} pred shape")
            terms.append(m.future_predictor.future_loss(pred, tgt))
        loss = flow_loss + 0.1 * torch.stack(terms).mean()
        loss.backward()
        self.assertIsNotNone(m.future_predictor.mlp[0].weight.grad, f"B={B}")
        self.assertFalse(torch.isnan(loss), f"B={B}")

    def test_future_chain_batch_neq_time(self) -> None:
        """B≠T（5/3/1）：裸索引 action_conditions[t] 在此形状下崩溃/错位。"""
        for B in (5, 3, 1):
            with self.subTest(B=B):
                self._run_future_chain(B)


class EvsmContractTests(unittest.TestCase):
    """EVSM：证据验证的暂存任务记忆（2026-08-07 Codex 主推）。

    - task_spec 暂存动作提议，下一周期经 future-latent 与真实视觉的
      一致性验证后才提交（q→1 提交 / q→0 回滚），task 字段始终是已提交态。
    - 训练链（B≠T）必须跑通：future loss 用 task_spec 而非 task。
    - gate 诊断统计随 VisualMemory 返回。
    """

    def _policy(self):
        cfg = VACompoundConfig(
            language_dim=12,
            vision_dim=10,
            hidden_dim=16,
            num_layers=4,
            num_heads=4,
            action_horizon=3,
            action_dim=4,
            proprio_dim=5,
            memory_split=True,
            evidence_tokens=6,
            task_tokens=4,
            sequential_coupling=2,
            flow_cond="adaln",
            future_predict=True,
            evsm=True,
            evsm_kappa=0.02,
            evsm_temp=0.005,
        )
        return VACompoundPolicy(cfg)

    def _fresh_policy_with_cache(self, B: int, lang: torch.Tensor, mask: torch.Tensor):
        m = self._policy().train()
        lc = m.build_language_cache(lang, mask)
        return m, lc

    def test_evsm_training_chain_batch_neq_time(self) -> None:
        """EVSM 完整训练链：B≠T 不崩溃，future loss 走 task_spec，梯度存在。"""
        for B in (5, 3, 1):
            with self.subTest(B=B):
                m = self._policy().train()
                T, H, A, V = 4, 3, 4, 10
                vis = torch.randn(B, T, 8, V)
                prop = torch.randn(B, T, 5)
                prev = torch.randn(B, T, A)
                lang = torch.randn(B, 4, 12)
                mask = torch.ones(B, 4, dtype=torch.bool)
                noisy = torch.randn(B, T, H, A)
                ftime = torch.rand(B, T)
                lc = m.build_language_cache(lang, mask)
                mem = None
                conds = []
                mems = []
                velos = []
                for t in range(T):
                    cond, mem = m.encode_condition(
                        vis[:, t], prop[:, t], prev[:, t],
                        language_cache=lc, visual_memory=mem, return_visual_memory=True,
                    )
                    conds.append(cond)
                    mems.append(mem)
                    velos.append(m.flow_velocity(cond, noisy[:, t], ftime[:, t]))
                action_conditions = torch.stack(conds, dim=1)
                flow_loss = torch.stack(velos).mean()
                terms = []
                for t in range(T - 1):
                    # EVSM：future 预测必须基于暂存提议（task_spec）
                    self.assertIsNotNone(mems[t].task_spec, f"B={B} t={t} task_spec")
                    pred = m.future_predictor(
                        action_conditions[:, t], mems[t].evidence, mems[t].task_spec
                    )
                    tgt = vis[:, t + 1].mean(dim=1)
                    terms.append(m.future_predictor.future_loss(pred, tgt))
                # 第二周期起有验证门控统计
                self.assertIsNotNone(mems[1].gate, f"B={B} gate at t=1")
                loss = flow_loss + 0.1 * torch.stack(terms).mean()
                loss.backward()
                self.assertFalse(torch.isnan(loss), f"B={B}")
                self.assertIsNotNone(
                    m.future_predictor.mlp[0].weight.grad, f"B={B} grad"
                )
                self.assertIsNotNone(
                    m.task_gate.gate.weight.grad, f"B={B} task_gate grad"
                )

    def test_evsm_commit_on_matching_evidence(self) -> None:
        """未来预测与真实视觉一致（δ≈0）→ q≈1 → 提议被提交（task≈spec）。"""
        B, H, A, V = 2, 3, 4, 10
        m, lc = self._fresh_policy_with_cache(
            B, torch.randn(B, 4, 12), torch.ones(B, 4, dtype=torch.bool)
        )
        vis = torch.randn(B, 8, V)
        prop = torch.randn(B, 5)
        prev = torch.randn(B, A)
        # 第一步：产生暂存提议
        _, mem = m.encode_condition(
            vis, prop, prev, language_cache=lc, return_visual_memory=True
        )
        self.assertIsNotNone(mem.task_spec)
        self.assertIsNotNone(mem.task)  # 语言初始化的工作区（commit 尚 None）
        self.assertIsNotNone(mem.pending_future)
        # 第二步：pending_future 与观察视觉高度一致 → 提交
        pf = mem.pending_future.detach()
        vis2 = (pf + 0.001 * torch.randn(B, V))[:, None, :].expand(-1, 8, -1)
        _, mem2 = m.encode_condition(
            vis2, prop, prev, language_cache=lc, visual_memory=mem, return_visual_memory=True
        )
        self.assertIsNotNone(mem2.task)
        # task = q*spec + (1-q)*prev_task；首周期 prev_task=None 时退化提交
        d_spec = (mem2.task - mem.task_spec.detach()).abs().mean().item()
        self.assertLess(d_spec, 0.5, f"matching evidence should commit; d={d_spec}")
        self.assertGreaterEqual(mem2.gate, 0.9, "matching evidence → gate high")

    def test_evsm_rollback_on_mismatching_evidence(self) -> None:
        """未来预测与真实视觉正交（δ≈1）→ q≈0 → 提议回滚（task≈prev_task）。"""
        B, H, A, V = 2, 3, 4, 10
        m, lc = self._fresh_policy_with_cache(
            B, torch.randn(B, 4, 12), torch.ones(B, 4, dtype=torch.bool)
        )
        vis = torch.randn(B, 8, V)
        prop = torch.randn(B, 5)
        prev = torch.randn(B, A)
        _, mem = m.encode_condition(
            vis, prop, prev, language_cache=lc, return_visual_memory=True
        )
        # 第二步：观察视觉与 pending_future 正交 → 回滚
        pf = mem.pending_future.detach()
        ortho = torch.randn(B, V)
        ortho = ortho - (ortho * pf).sum(-1, keepdim=True) * pf / (
            pf.norm(dim=-1, keepdim=True) ** 2 + 1e-8
        )
        vis2 = ortho[:, None, :].expand(-1, 8, -1)
        _, mem2 = m.encode_condition(
            vis2, prop, prev, language_cache=lc, visual_memory=mem, return_visual_memory=True
        )
        self.assertIsNotNone(mem2.task)
        self.assertLess(mem2.gate, 0.05, "orthogonal evidence → gate low")
        # 回滚：commit = q*spec + (1-q)*prev_task ≈ prev_task（T_init）
        d_prev = (mem2.task - mem.task.detach()).abs().mean().item()
        self.assertLess(d_prev, 0.5, f"orthogonal evidence should roll back; d={d_prev}")
        self.assertFalse(torch.isnan(mem2.task).any())
