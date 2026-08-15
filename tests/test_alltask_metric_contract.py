from __future__ import annotations

import inspect
from functools import cache
from types import SimpleNamespace

import numpy as np
import torch

import prepare_metaworld_metric as metric_data

from prepare_metaworld_metric import (
    DIRECT_TOOL_TARGET_TASKS,
    SUPPORTED_TASKS,
    TASK_ALIGNED_ROLE_SOURCES,
    _chronological_capture_offsets,
    _entity_aware_visibility,
    _randomize_nonrobot_articulation,
    _relation_labels,
    _resize_chronological_frames,
    _role_visibility,
    _sample_one,
    keypoint_world_positions,
    make_metric_batch,
)
from scripts.build_longtraj_features import ENV_TO_TASK
from train_metric_visual import (
    _alias_consistency_loss,
    _pair_geometry_consistency_loss,
    compute_losses,
)


@cache
def _benchmark_and_tasks():
    import metaworld

    benchmark = metaworld.MT50(seed=17)
    task_by_name = {}
    for task in benchmark.train_tasks:
        task_by_name.setdefault(task.env_name, task)
    return benchmark, task_by_name


def _mt50_environments():
    benchmark, task_by_name = _benchmark_and_tasks()
    for task_name in SUPPORTED_TASKS:
        env = benchmark.train_classes[task_name]()
        env.set_task(task_by_name[task_name])
        np.random.seed(120812)
        env.reset()
        yield task_name, env


def test_metric_frames_use_action_runtime_chronological_order() -> None:
    assert _chronological_capture_offsets(4) == (0, 2, 4, 6)
    markers = [
        np.full((8, 8, 3), value, dtype=np.uint8)
        for value in (10, 20, 30, 40)
    ]
    resized = _resize_chronological_frames(markers)
    assert [int(frame[0, 0, 0]) for frame in resized] == [10, 20, 30, 40]


def test_each_metric_sample_randomizes_articulation_once() -> None:
    source = inspect.getsource(_sample_one)
    assert source.count("_randomize_nonrobot_articulation(env, rng)") == 1


def test_all_49_tasks_follow_declared_role_contracts() -> None:
    assert SUPPORTED_TASKS == tuple(ENV_TO_TASK)
    assert len(SUPPORTED_TASKS) == 49
    assert TASK_ALIGNED_ROLE_SOURCES == {
        "peg-insert-side-v3": ("tcp_center", "pegGrasp", "hole", "pegHead")
    }

    checked = []
    for task, env in _mt50_environments():
        try:
            world = keypoint_world_positions(env, task)
            assert world is not None
            assert world.shape == (4, 3)
            assert np.isfinite(world).all()
            np.testing.assert_allclose(world[0], env.tcp_center)
            if task == "peg-insert-side-v3":
                np.testing.assert_allclose(world[1], env.data.site("pegGrasp").xpos)
                np.testing.assert_allclose(world[2], env.data.site("hole").xpos)
                np.testing.assert_allclose(world[3], env.data.site("pegHead").xpos)
                assert not np.allclose(world[1], world[3])
                assert not np.allclose(world[2], env._target_pos)
            else:
                np.testing.assert_allclose(world[2], env._target_pos)
                entities = np.asarray(env._get_pos_objects()).reshape(-1, 3)
                if task in DIRECT_TOOL_TARGET_TASKS:
                    np.testing.assert_allclose(world[1], world[0])
                    np.testing.assert_allclose(world[3], world[0])
                else:
                    np.testing.assert_allclose(world[1], entities[0])
                    expected_progress = entities[1] if len(entities) > 1 else entities[0]
                    np.testing.assert_allclose(world[3], expected_progress)
            checked.append(task)
        finally:
            env.close()
    assert tuple(checked) == SUPPORTED_TASKS


def test_task35_aligned_roles_are_observable_and_distinct() -> None:
    batch = make_metric_batch(
        "peg-insert-side-v3", np.random.default_rng(35035), 32
    )
    # role order: [tool, pegGrasp, hole(target), pegHead(interface)].  The old
    # contract produced target visibility 0 and object==interface; both are P0.
    assert float(batch["visibility"][:, 2].mean()) >= 0.5
    assert float(batch["visibility"][:, 3].mean()) >= 0.5
    separation = np.linalg.norm(batch["world"][:, 1] - batch["world"][:, 3], axis=-1)
    assert float(separation.min()) > 0.05
    relation_error = batch["keypoints"][:, 3] - batch["keypoints"][:, 2]
    np.testing.assert_allclose(
        batch["relation"][:, 2:4], relation_error, atol=1e-7, rtol=1e-6
    )
    assert batch["meta"]["task_role_source"]["task_overrides"] == {
        "peg-insert-side-v3": ["tcp_center", "pegGrasp", "hole", "pegHead"]
    }


def test_dual_entity_and_direct_tcp_families_have_correct_relations() -> None:
    kp = np.array(
        [[0.1, 0.1], [0.2, 0.3], [0.8, 0.9], [0.6, 0.7]],
        dtype=np.float32,
    )
    world = np.array(
        [[0.0, 0.0, 0.4], [0.1, 0.2, 0.3], [0.8, 0.9, 0.1], [0.6, 0.7, 0.2]],
        dtype=np.float32,
    )
    relation, aux = _relation_labels(kp, world)
    np.testing.assert_allclose(relation[:2], kp[0] - kp[1])
    np.testing.assert_allclose(relation[2:4], kp[3] - kp[2])
    np.testing.assert_allclose(relation[5], world[3, 2] - world[2, 2])
    np.testing.assert_allclose(aux[2], np.linalg.norm(world[0] - world[1]))
    np.testing.assert_allclose(aux[3], np.linalg.norm(world[3] - world[2]))


def test_virtual_in_frame_target_is_not_used_as_visual_supervision() -> None:
    pixels = np.full((4, 2), 100.0)
    point_depths = np.ones(4)
    surface_depth = np.full((480, 480), 4.0, dtype=np.float32)
    visibility, surface_visible, in_frame = _role_visibility(
        pixels, point_depths, surface_depth
    )
    np.testing.assert_array_equal(surface_visible, np.zeros(4))
    np.testing.assert_array_equal(visibility, np.zeros(4))
    np.testing.assert_array_equal(in_frame, np.ones(4))


def test_internal_drawer_and_peg_anchors_use_same_entity_first_hit(monkeypatch) -> None:
    world = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.8, 0.8, 0.8], [0.4, 0.5, 0.6]]
    )
    monkeypatch.setattr(metric_data, "_entity_anchor_body", lambda env, point: 11)
    monkeypatch.setattr(metric_data, "_first_ray_hit_body", lambda env, point: 11)
    monkeypatch.setattr(metric_data, "_is_robot_body", lambda env, body: False)
    entity_visible = _entity_aware_visibility(
        object(), world, np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)
    )
    # Only the generic entity roles are relaxed; tool and target remain strict.
    np.testing.assert_array_equal(entity_visible, np.array([0, 1, 0, 1]))


def test_entity_visibility_does_not_accept_other_entity_or_robot_occluder(monkeypatch) -> None:
    world = np.arange(12, dtype=float).reshape(4, 3)
    monkeypatch.setattr(metric_data, "_entity_anchor_body", lambda env, point: 11)
    monkeypatch.setattr(metric_data, "_first_ray_hit_body", lambda env, point: 12)
    monkeypatch.setattr(metric_data, "_is_robot_body", lambda env, body: body == 12)
    hidden = _entity_aware_visibility(
        object(), world, np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)
    )
    np.testing.assert_array_equal(hidden, np.zeros(4))


def test_real_drawer_internal_anchor_gets_entity_visibility() -> None:
    batch = make_metric_batch(
        "drawer-close-v3", np.random.default_rng(777), 2
    )
    # The official drawer achieved-position projects near the handle silhouette,
    # not onto its depth surface.  Generic same-body neighborhood rays recover it.
    assert float(batch["surface_visible"][:, 1].sum()) == 0.0
    assert float(batch["surface_visible"][:, 3].sum()) == 0.0
    assert float(batch["entity_visible"][:, 1].sum()) > 0.0
    assert float(batch["entity_visible"][:, 3].sum()) > 0.0


def test_alias_consistency_only_acts_on_same_entity_coordinates() -> None:
    keypoints = torch.tensor(
        [
            [[0.1, 0.1], [0.4, 0.4], [0.8, 0.8], [0.4, 0.4]],
            [[0.1, 0.1], [0.3, 0.3], [0.8, 0.8], [0.7, 0.7]],
        ]
    )

    def output(component: str | None, sample: int):
        p = torch.zeros(2, 4, 2)
        logits = torch.zeros(2, 4)
        scores = torch.zeros(2, 4, 12)
        if component == "position":
            p[sample, 3] = 1.0
        elif component == "visibility":
            logits[sample, 3] = 1.0
        elif component == "scores":
            scores[sample, 3] = 1.0
        return SimpleNamespace(p=p, visibility_logits=logits, scores=scores)

    for component in ("position", "visibility", "scores"):
        alias_loss, count = _alias_consistency_loss(
            output(component, 0), keypoints
        )
        ignored_loss, ignored_count = _alias_consistency_loss(
            output(component, 1), keypoints
        )
        assert count == ignored_count == 1
        assert alias_loss > 0, component
        torch.testing.assert_close(ignored_loss, torch.tensor(0.0))


def test_pair_geometry_penalizes_mode_hop_and_masks_hidden_endpoints() -> None:
    keypoints = torch.tensor(
        [[[0.20, 0.20], [0.30, 0.30], [0.80, 0.80], [0.60, 0.60]]]
    )
    correct = SimpleNamespace(p=keypoints.clone())
    mode_hop = SimpleNamespace(p=keypoints.clone())
    mode_hop.p[:, 0] = torch.tensor([0.90, 0.10])
    visible = torch.ones(1, 4)
    correct_loss, correct_n = _pair_geometry_consistency_loss(
        correct, keypoints, visible
    )
    wrong_loss, wrong_n = _pair_geometry_consistency_loss(
        mode_hop, keypoints, visible
    )
    assert correct_n == wrong_n == 2
    torch.testing.assert_close(correct_loss, torch.tensor(0.0))
    assert wrong_loss > correct_loss

    tool_hidden = visible.clone()
    tool_hidden[:, 0] = 0.0
    masked_loss, masked_n = _pair_geometry_consistency_loss(
        mode_hop, keypoints, tool_hidden
    )
    assert masked_n == 1
    torch.testing.assert_close(masked_loss, torch.tensor(0.0))


def test_hidden_relation_endpoints_do_not_contribute_visual_loss() -> None:
    keypoints = torch.full((1, 4, 2), 0.5)
    visibility = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    target_relation = torch.zeros(1, 6)
    common = {
        "p": keypoints.clone(),
        "log_heatmap": torch.full((1, 4, 24, 24), -np.log(24 * 24)),
        "visibility_logits": torch.zeros(1, 4),
    }
    clean = SimpleNamespace(**common, relation=torch.zeros(1, 6))
    hidden_wrong = SimpleNamespace(
        **common,
        relation=torch.tensor([[0.0, 0.0, 100.0, -100.0, 100.0, -100.0]]),
    )
    clean_loss, _ = compute_losses(
        clean, keypoints, visibility, target_relation, loc_only=False
    )
    hidden_loss, _ = compute_losses(
        hidden_wrong, keypoints, visibility, target_relation, loc_only=False
    )
    torch.testing.assert_close(hidden_loss, clean_loss)


def test_non_loc_only_offset_supervision_contributes_to_total_loss() -> None:
    keypoints = torch.tensor([[[0.52, 0.52]] * 4])
    visibility = torch.ones(1, 4)
    relation = torch.zeros(1, 6)
    grid = 24
    common = {
        "p": keypoints.clone(),
        "log_heatmap": torch.full((1, 4, grid, grid), -np.log(grid * grid)),
        "scores": torch.zeros(1, 4, 2 * grid * grid),
        "visibility_logits": torch.full((1, 4), 20.0),
        "relation": relation.clone(),
    }
    zero_offset = SimpleNamespace(
        **common, offset_full=torch.zeros(1, 4, 2 * grid * grid, 2)
    )
    wrong_offset = SimpleNamespace(
        **common, offset_full=torch.ones(1, 4, 2 * grid * grid, 2)
    )
    loss_without, _ = compute_losses(
        wrong_offset, keypoints, visibility, relation,
        loc_only=False, offset_supervision=False,
    )
    loss_zero, parts_zero = compute_losses(
        zero_offset, keypoints, visibility, relation,
        loc_only=False, offset_supervision=True,
    )
    loss_wrong, parts_wrong = compute_losses(
        wrong_offset, keypoints, visibility, relation,
        loc_only=False, offset_supervision=True,
    )
    assert "offset" in parts_zero and "offset" in parts_wrong
    assert {"alias", "alias_weighted", "alias_n"} <= set(parts_zero)
    assert loss_wrong > loss_zero
    torch.testing.assert_close(
        loss_wrong - loss_without,
        torch.tensor(parts_wrong["offset"]),
    )


def test_articulation_randomizer_is_task_agnostic() -> None:
    source = inspect.getsource(_randomize_nonrobot_articulation)
    assert "door-lock-v3" not in source

    selected = {"button-press-v3", "door-lock-v3", "window-open-v3"}
    seen = set()
    for task, env in _mt50_environments():
        try:
            if task not in selected:
                continue
            before = env.data.qpos.copy()
            count = _randomize_nonrobot_articulation(
                env, np.random.default_rng(100 + len(seen))
            )
            assert count >= 1
            assert not np.allclose(before, env.data.qpos)
            seen.add(task)
        finally:
            env.close()
    assert seen == selected
