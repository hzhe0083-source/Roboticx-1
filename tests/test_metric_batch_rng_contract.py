"""CPU-only tests for batch-invariant online metric sample generation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import prepare_metaworld_metric as metric_data


class _FakeModel:
    def __init__(self) -> None:
        self.geom_rgba = np.array(
            [[0.8, 0.4, 0.2, 1.0], [0.1, 0.3, 0.7, 1.0]], dtype=np.float64
        )


class _FakeEnv:
    def __init__(self, task: str) -> None:
        self.task = task
        self.model = _FakeModel()
        self.reset_draw = -1

    def reset(self, *, seed: int) -> None:
        # Model colours intentionally survive reset, as they do in MuJoCo.
        self.reset_draw = int(np.random.randint(0, 2**31))

    def close(self) -> None:
        pass


class _FakeRenderer:
    def __init__(self, env: _FakeEnv) -> None:
        self.env = env

    def close(self) -> None:
        pass


def _fake_make_env(task: str, seed: int) -> _FakeEnv:
    env = _FakeEnv(task)
    env.reset(seed=seed)
    return env


def _fake_sample(
    env: _FakeEnv,
    task: str,
    rng: np.random.Generator,
    w: int,
    *,
    include_raw_frames: bool = False,
    renderer=None,
) -> dict:
    # Exercise all three state sources used by the real generator: env reset,
    # isolated Generator draws and legacy global np.random draws.  Mutating
    # geom_rgba catches state leakage when one env is reused inside a batch.
    local = rng.random(12)
    legacy = float(np.random.random())
    colour_before = env.model.geom_rgba[:, :3].copy()
    env.model.geom_rgba[:, :3] *= 0.5 + rng.random()
    base = np.array(
        [env.reset_draw / float(2**31), legacy, colour_before[0, 0], local[0]],
        dtype=np.float32,
    )
    keypoints = np.stack((base, base + 0.01), axis=1)
    visibility = (local[4:8] > 0.5).astype(np.float32)
    world = local.reshape(4, 3).astype(np.float32)
    record = {
        "frames": np.full((w, 384, 384, 3), int(local[0] * 255), dtype=np.uint8),
        "keypoints": keypoints,
        "visibility": visibility,
        "surface_visible": visibility.copy(),
        "entity_visible": visibility.copy(),
        "in_frame": np.ones(4, dtype=np.float32),
        "relation": local[:6].astype(np.float32),
        "relation_aux": local[:4].astype(np.float32),
        "contact": np.float32(local[0] > 0.5),
        "world": world,
        "supported": True,
    }
    if include_raw_frames:
        record["raw_frames"] = np.full(
            (w, 480, 480, 3), int(local[1] * 255), dtype=np.uint8
        )
    return record


def _concatenate(chunks: list[dict], key: str):
    if key == "tasks":
        return [task for chunk in chunks for task in chunk[key]]
    return np.concatenate([np.asarray(chunk[key]) for chunk in chunks], axis=0)


@pytest.mark.parametrize("task", [metric_data.SUPPORTED_TASKS[0], "any"])
def test_once_eight_matches_two_calls_of_four(monkeypatch, task: str) -> None:
    monkeypatch.setattr(metric_data, "make_env", _fake_make_env)
    monkeypatch.setattr(metric_data, "_Renderer", _FakeRenderer)
    monkeypatch.setattr(metric_data, "_sample_one", _fake_sample)

    once_rng = np.random.default_rng(20260812)
    once = metric_data.make_metric_batch(
        task, once_rng, 8, include_raw_frames=True
    )
    once_next = once_rng.integers(0, 2**63)

    split_rng = np.random.default_rng(20260812)
    split = [
        metric_data.make_metric_batch(
            task, split_rng, 4, include_raw_frames=True
        ),
        metric_data.make_metric_batch(
            task, split_rng, 4, include_raw_frames=True
        ),
    ]
    split_next = split_rng.integers(0, 2**63)

    for key in (
        "frames",
        "raw_frames",
        "keypoints",
        "visibility",
        "surface_visible",
        "entity_visible",
        "in_frame",
        "relation",
        "relation_aux",
        "contact",
        "world",
        "supported",
    ):
        np.testing.assert_array_equal(once[key], _concatenate(split, key), err_msg=key)
    assert once["tasks"] == _concatenate(split, "tasks")
    assert once_next == split_next
    assert once["meta"]["sample_rng_contract"] == metric_data.SAMPLE_RNG_CONTRACT


def test_metric_batch_rejects_empty_sample_request() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        metric_data.make_metric_batch(
            metric_data.SUPPORTED_TASKS[0], np.random.default_rng(0), 0
        )


def test_fixed_task_reuses_one_renderer_and_closes_at_batch_boundary(monkeypatch) -> None:
    events: list[tuple[str, object]] = []

    class FakeRenderer:
        def __init__(self, env) -> None:
            self.env = env
            events.append(("construct", env))

        def close(self) -> None:
            events.append(("close", self.env))

    seen_renderers = []

    def fake_sample(env, task, rng, w, *, include_raw_frames=False, renderer=None):
        assert renderer is not None
        seen_renderers.append(renderer)
        events.append(("sample", renderer))
        return _fake_sample(
            env,
            task,
            rng,
            w,
            include_raw_frames=include_raw_frames,
            renderer=renderer,
        )

    monkeypatch.setattr(metric_data, "make_env", _fake_make_env)
    monkeypatch.setattr(metric_data, "_Renderer", FakeRenderer)
    monkeypatch.setattr(metric_data, "_sample_one", fake_sample)

    metric_data.make_metric_batch(
        metric_data.SUPPORTED_TASKS[0], np.random.default_rng(17), 5
    )

    assert len(seen_renderers) == 5
    assert len({id(renderer) for renderer in seen_renderers}) == 1
    assert [kind for kind, _ in events] == [
        "construct",
        "sample",
        "sample",
        "sample",
        "sample",
        "sample",
        "close",
    ]


def test_sample_one_does_not_close_injected_renderer(monkeypatch) -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.data = SimpleNamespace(
                mocap_pos=np.zeros((1, 3), dtype=np.float64),
                mocap_quat=np.zeros((1, 4), dtype=np.float64),
            )
            self.actions = []

        def step(self, action) -> None:
            self.actions.append(action.copy())

    class InjectedRenderer:
        def __init__(self) -> None:
            self.rgb_calls = 0
            self.depth_calls = 0
            self.close_calls = 0

        def render_rgb(self):
            self.rgb_calls += 1
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def render_depth(self):
            self.depth_calls += 1
            return np.ones((8, 8), dtype=np.float32)

        def close(self) -> None:
            self.close_calls += 1

    for name in (
        "_randomize_colors",
        "_jitter_camera",
        "_move_object_random",
        "_randomize_nonrobot_articulation",
    ):
        monkeypatch.setattr(metric_data, name, lambda env, rng: None)
    monkeypatch.setattr(metric_data, "_import_mujoco_metaworld", lambda: (object(), object()))
    monkeypatch.setattr(
        metric_data,
        "keypoint_world_positions",
        lambda env, task: np.zeros((4, 3), dtype=np.float64),
    )
    monkeypatch.setattr(
        metric_data,
        "project_points",
        lambda env, world: (
            np.full((4, 2), 4.0, dtype=np.float64),
            np.ones(4, dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        metric_data,
        "_role_visibility",
        lambda pixels, depths, depth: tuple(
            np.ones(4, dtype=np.float32) for _ in range(3)
        ),
    )
    monkeypatch.setattr(
        metric_data,
        "_entity_aware_visibility",
        lambda env, world, surface_visible, in_frame: surface_visible.copy(),
    )

    renderer = InjectedRenderer()
    record = metric_data._sample_one(
        FakeEnv(),
        metric_data.SUPPORTED_TASKS[0],
        np.random.default_rng(41),
        2,
        renderer=renderer,
    )

    assert record["frames"].shape == (2, 384, 384, 3)
    assert renderer.rgb_calls == 2
    assert renderer.depth_calls == 1
    assert renderer.close_calls == 0


def test_mixed_task_keeps_sample_owned_renderer_behavior(monkeypatch) -> None:
    specs = [
        (metric_data.SUPPORTED_TASKS[0], 11),
        (metric_data.SUPPORTED_TASKS[1], 22),
        (metric_data.SUPPORTED_TASKS[0], 33),
    ]
    seen_renderers = []

    def fake_sample(env, task, rng, w, *, include_raw_frames=False, renderer=None):
        seen_renderers.append(renderer)
        return _fake_sample(
            env,
            task,
            rng,
            w,
            include_raw_frames=include_raw_frames,
            renderer=renderer,
        )

    monkeypatch.setattr(metric_data, "_derive_sample_specs", lambda task, rng, n: specs)
    monkeypatch.setattr(metric_data, "make_env", _fake_make_env)
    monkeypatch.setattr(metric_data, "_sample_one", fake_sample)

    metric_data.make_metric_batch("any", np.random.default_rng(23), len(specs))

    assert seen_renderers == [None, None, None]
