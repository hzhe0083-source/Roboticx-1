"""eval_metaworld.py 归一化→clip→反归一化契约回归测试（纯 CPU，不 import metaworld）。

对照代码（eval_metaworld.py 决策循环）：
  norm_action = np.clip(chunk[(step - chunk_start_step) % ACTION_HORIZON], -1.0, 1.0)
  action = norm_action * (aq99 - aq01) / 2 + (aq99 + aq01) / 2
  last_norm = norm_action                       # prev 反馈使用裁剪后值
  state = np.clip(2.0 * (obs[:4] - sq01) / scale_s - 1.0, -1.0, 1.0)
  scale_s = np.where(np.abs(sq99 - sq01) < 1e-6, 1.0, sq99 - sq01)

契约点：
1. 模型输出的归一化动作可能越出 [-1,1]（flow 采样无硬约束），必须先 clip 再反归一化；
   裁剪后反归一化值 == clip 后值经 aq01/aq99 的算术映射。
2. prev 反馈（last_norm）必须用裁剪后的值，不能用模型原始输出——训练标签存盘即 clip。
"""
from pathlib import Path

import numpy as np
import pytest
import torch

DATA = Path(__file__).resolve().parents[1] / "data" / "metaworld_features_v4.pt"


@pytest.fixture(scope="module")
def norm():
    if not DATA.exists():
        pytest.skip(f"missing {DATA}")
    d = torch.load(DATA, map_location="cpu", weights_only=False)
    n = d["normalization"]
    return {
        "aq01": n["action_q01"].numpy(),
        "aq99": n["action_q99"].numpy(),
        "sq01": n["state_q01"].numpy(),
        "sq99": n["state_q99"].numpy(),
    }


def denorm(norm_action: np.ndarray, aq01: np.ndarray, aq99: np.ndarray) -> np.ndarray:
    """与 eval_metaworld.py 逐字一致的动作反归一化。"""
    return norm_action * (aq99 - aq01) / 2 + (aq99 + aq01) / 2


def test_out_of_range_norm_action_clipped_before_denorm(norm):
    """越界归一化动作（1.7, -1.3）反归一化前必须被 clip 到 [-1,1]。"""
    aq01, aq99 = norm["aq01"], norm["aq99"]
    raw = np.array([1.7, -1.3, 0.5, -0.9])  # 前两维越界，后两维在界内
    clipped = np.clip(raw, -1.0, 1.0)
    np.testing.assert_array_equal(clipped, np.array([1.0, -1.0, 0.5, -0.9]))

    action = denorm(clipped, aq01, aq99)
    expected = clipped * (aq99 - aq01) / 2 + (aq99 + aq01) / 2
    np.testing.assert_allclose(action, expected)

    # clip 必须发生在反归一化之前：否则越界维度的动作会超出训练范围
    # （denorm(1.7) > q99、denorm(-1.3) < q01），界内维度不受影响
    unclipped = denorm(raw, aq01, aq99)
    assert action[0] < unclipped[0]  # 1.7 被拉回 1.0 → 动作变小
    assert action[1] > unclipped[1]  # -1.3 被拉回 -1.0 → 动作变大
    assert action[2] == pytest.approx(unclipped[2])  # 0.5 在界内，不变
    assert action[3] == pytest.approx(unclipped[3])  # -0.9 在界内，不变


def test_clipped_action_maps_exactly_to_q01_q99_bounds(norm):
    """裁剪后归一化动作的映射边界 == q01/q99（训练数据原始动作范围）。"""
    aq01, aq99 = norm["aq01"], norm["aq99"]
    np.testing.assert_allclose(denorm(np.full(4, 1.0), aq01, aq99), aq99)
    np.testing.assert_allclose(denorm(np.full(4, -1.0), aq01, aq99), aq01)
    np.testing.assert_allclose(denorm(np.zeros(4), aq01, aq99), (aq99 + aq01) / 2)


def test_prev_feedback_uses_clipped_value(norm):
    """模拟决策循环：last_norm（下次决策的 previous_action 输入）必须是裁剪后值。"""
    aq01, aq99 = norm["aq01"], norm["aq99"]
    # 每一步 chunk 行都可能越界；最后一行刻意含越界值
    chunk = np.array(
        [
            [1.7, -1.3, 0.5, -0.9],
            [0.2, -0.4, 1.2, 0.0],
            [1.2, 0.3, -1.4, 0.1],
        ]
    )
    last_norm = np.zeros(4)
    for row in chunk:
        norm_action = np.clip(row, -1.0, 1.0)
        _action = denorm(norm_action, aq01, aq99)  # 执行环境步（这里不真正 step）
        last_norm = norm_action  # eval_metaworld.py: last_norm = norm_action

    clipped = np.clip(chunk[-1], -1.0, 1.0)
    np.testing.assert_array_equal(last_norm, clipped)
    assert not np.array_equal(last_norm, chunk[-1])  # 越界行不能原样进 prev
    assert last_norm[0] == 1.0 and last_norm[2] == -1.0
    # 界内维保持不变
    assert last_norm[1] == pytest.approx(chunk[-1][1])
    assert last_norm[3] == pytest.approx(chunk[-1][3])


def test_state_norm_clipped_to_unit_box(norm):
    """状态归一化同样 clip 到 [-1,1]（eval_metaworld.py 的 state 路径）。"""
    sq01, sq99 = norm["sq01"], norm["sq99"]
    scale_s = np.where(np.abs(sq99 - sq01) < 1e-6, 1.0, sq99 - sq01)
    obs = np.array([-5.0, 0.5, 3.0, 0.3])  # dim0 越下界、dim2 越上界
    state = np.clip(2.0 * (obs - sq01) / scale_s - 1.0, -1.0, 1.0)

    assert state[0] == -1.0 and state[2] == 1.0
    linear = 2.0 * (obs - sq01) / scale_s - 1.0
    assert state[1] == pytest.approx(linear[1])  # 界内维线性映射不变
    assert state[3] == pytest.approx(linear[3])
