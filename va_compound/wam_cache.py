"""E7 WAM v1 — cache 构建器 + episode split + last-slice 池化（M0，CPU-only）。

本文件是 Task 5 的唯一所有权文件；其他 agent（Task 3 rollout hook、Task 6
trainer、Task 7 probe、Task 8 tests）按文末契约调用。

内容：
  1. ``wam_split_from_episode``：episode_id % 10 → 0..7 train、8 val、9 test；
     整条 episode 归属一个 split，禁止随机窗口切分。
  2. ``wam_anchor_index`` / ``wam_future_frame`` / ``future_frame_in_bounds``：
     4 决策上下文锚点索引与未来帧换算。k 单位 = 动作列/帧偏移
     （action_axis_units="frame_offset"），帧号 = decision_frame + k。
  3. ``wam_last_slice_pool``：H11 dense [B,1152,768] → 最后时间片 576 →
     24×24 网格 → 6×6 块均值 → 4×4=16 空间 token [B,16,768]。刻意不用
     ``pool_mtvj_coarse_tokens``（那是 16 个连续桶，不是 4×4 空间池化）。
  4. ``build_wam_cache``：流式（按 task 分组）读 windows 文件，每 anchor
     一条记录，按 task 分片 torch.save + manifest.json + index.json。
  5. ``WAMCacheDataset``：按 split mask 读取分片记录，__getitem__ 返回
     record dict。
  6. ``assert_record_schema``：record 键/类型/形状白名单校验（RECORD_SCHEMA），
     cache 写分片前与 trainer collate 前共用，杜绝两侧 schema 漂移。

M0 阶段 latent/condition 编码依赖冻结基座前向（GPU 被基座训练占用），因此
这些字段以占位零张量写入、形状与最终一致，由 M1 GPU 管线调用
``fill_record_latents(record, base_outputs)`` 填充后重写分片。
``windows_pt=None`` 即 --fake 合成模式：生成合成小数据集并走完整管线
（含合成 latent），CPU 端到端可测。

输入 windows 文件契约（data/metaworld_longtraj_windows_h48_all49_repaired_v2.pt）：
  actions [N,4,48,4]（q01/q99 归一化）、action_valid_mask [N,4,48] bool、
  recovery_mask [N,4,48] bool、episode_id [N]、instruction_id [N]、
  frame_refs [(task_file, ep_idx, frame_idx[4,4])]（frame_idx 最后一列 =
  该决策点的帧索引，帧/状态 i 为 action[i] 执行前观测）。

每 anchor 一条记录（按 task 分片存储；键白名单 = RECORD_SCHEMA，cache 写入与
trainer 读取两侧都经 ``assert_record_schema`` 校验，见「record schema 白名单」节）：
  action_condition [48,512]        VA action condition（M1 填）
  va_layers        8×[16,512]      VA 记忆快照（M1 填）
  spatial16        [16,768]        H11 最后时间片 4×4 池化 = z(d)（M1 填）
  geo8             [8]             g(d) = p*visibility 扁平（M1 填）
  actions          [48,4]          q01/q99 归一化动作（M0 写）；action[h] 对应
                                   帧 decision_frame + h（动作列索引 = 帧偏移）
  future_latent_target [3,16,768]  Δlatent = z(d+k) − z(d)，k∈(6,24,48)
                                   （M1 填；M0 fake 同语义）
  future_geo_target    [3,2,8]     slice0 = g_future = p*vis(d+k)（绝对），
                                   slice1 = ν = g_future − g_current
                                   （M1 填；M0 fake 同语义）
  action_valid     [48]            动作监督 mask（M0 写）
  valid_future     [3]             未来帧越界 mask（M0 写，语义见下）
  perturbed_future [3]             未来目标排除 mask（M0 写，语义见下）
  horizon_weight   [3]             (1.0, 0.5, 0.25)（M0 写）
  episode_id / task_id             int（M0 写）
  task_file / ep_idx / decision_frame  M1 填 latent 所需的 provenance（M0 写）

perturbed_future[h]=True 表示第 h 个跨度未来目标应从损失中排除，当且仅当：
  - 未来窗口 [d+k-6, d+k-4, d+k-2, d+k]（列裁剪到 0..47）内任一帧
    action_valid_mask 为 False。action_valid_mask 已由 phase1 折叠了
    所有排除语义（build_longtraj_features.py:513-518）：
    * 成功后（valid[first_success+1:]=False）、settle、无效动作；
    * 「从正常阶段跨越外部随机扰动」= recovery 目标 ∧ 决策点早于
      perturb_start（unseen_recovery）——扰动已可观测的 recovery 监督
      有意保留。因此直接 OR recovery_mask 会误伤 ~80% 目标（recovery 段
      从扰动持续到 first_success），这里只信 action_valid_mask。

valid_future[h]=False 表示未来帧 d+k 超出该 episode 帧范围（+48 终点
frame d+48 不在 48 个动作列内，perturbed_future 的窗口检查够不到它，
由 ``future_frame_in_bounds(decision_frame, k, last_frame)`` 单独判定）。
per-task 文件存在时按每 episode 帧数写；文件缺失或 --fake 模式给合理
默认 True（M1 有 raw episode 时须用真实 last_frame 重填并自查）。
trainer 端组合权重 w = horizon_weight · (~perturbed_future) · valid_future，
无效跨度退出损失分母。
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent

# ---- 契约常量（与 E7 基座事实对齐，2026-08-13） ----
CONTRACT = "e7_wam_cache_v1"
CONTRACT_VERSION = 1
ACTION_AXIS_UNITS = "frame_offset"   # 动作列索引 = 帧偏移：action[h] 对应 decision_frame + h
HORIZONS = (6, 24, 48)          # 未来跨度（帧偏移/动作列单位）
HORIZON_WEIGHTS = (1.0, 0.5, 0.25)
N_VA_LAYERS = 8                 # VA 层数（memory_split=False → 每层 [B,16,512]）
VA_TOKENS = 16
VA_DIM = 512
VISION_DIM = 768               # V-JEPA H11 latent 维度
GEO_DIM = 8                    # MT-VJ p_times_visibility_flat
N_SCENE_TOKENS = 16            # 4×4 空间池化 token 数
H11_DENSE_TOKENS = 1152        # 2 时间片 × 24×24
LAST_SLICE_TOKENS = 576        # 最后（最新）时间片 = 24×24
H11_SPATIAL_GRID = 24
POOL_GRID = 4
POOL_BLOCK = H11_SPATIAL_GRID // POOL_GRID  # 6
ACTION_HORIZON = 48
ACTION_DIM = 4
SEQUENCE_LENGTH = 4            # 决策上下文窗口决策点数
CONTROL_STRIDE = 6             # 相邻决策点帧间隔
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------
# record schema 白名单（cache 写入与 trainer 读取的唯一契约）
# --------------------------------------------------------------------------

RECORD_SCHEMA: dict = {
    "action_condition": ("float", (ACTION_HORIZON, VA_DIM)),
    "va_layers": ("va_layers", (N_VA_LAYERS, VA_TOKENS, VA_DIM)),
    "spatial16": ("float", (N_SCENE_TOKENS, VISION_DIM)),
    "geo8": ("float", (GEO_DIM,)),
    "actions": ("float", (ACTION_HORIZON, ACTION_DIM)),
    "future_latent_target": ("float", (3, N_SCENE_TOKENS, VISION_DIM)),
    "future_geo_target": ("float", (3, 2, GEO_DIM)),
    "action_valid": ("bool", (ACTION_HORIZON,)),
    "perturbed_future": ("bool", (3,)),
    "valid_future": ("bool", (3,)),
    "horizon_weight": ("float", (3,)),
    "episode_id": ("int", None),
    "task_id": ("int", None),
    "task_file": ("str", None),
    "ep_idx": ("int", None),
    "decision_frame": ("int", None),
}


def assert_record_schema(record: dict) -> dict:
    """按 RECORD_SCHEMA 白名单校验 record 的键/类型/形状；违反即 ValueError。

    cache 构建器在写分片前调用，trainer 在 collate 每个 record 前调用，
    保证两侧使用同一组键与同一语义（严格白名单：缺键/多余键都报错）。
    通过则原样返回 record。
    """
    if not isinstance(record, dict):
        raise ValueError(f"record 必须是 dict，got {type(record).__name__}")
    expected = set(RECORD_SCHEMA)
    actual = set(record)
    if actual != expected:
        raise ValueError(
            f"record 键与白名单不符：missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    for key, (kind, shape) in RECORD_SCHEMA.items():
        value = record[key]
        if kind == "va_layers":
            if not isinstance(value, (list, tuple)) or len(value) != N_VA_LAYERS:
                n = len(value) if isinstance(value, (list, tuple)) else None
                raise ValueError(
                    f"{key} 必须是 {N_VA_LAYERS} 层 list/tuple，got "
                    f"{type(value).__name__}/{n}"
                )
            for i, layer in enumerate(value):
                if (not isinstance(layer, torch.Tensor)
                        or tuple(layer.shape) != (VA_TOKENS, VA_DIM)):
                    raise ValueError(
                        f"{key}[{i}] 形状 {getattr(layer, 'shape', None)} != "
                        f"({VA_TOKENS},{VA_DIM})"
                    )
        elif kind == "float":
            if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
                raise ValueError(
                    f"{key} 必须是浮点 torch.Tensor，got {type(value).__name__}"
                )
            if tuple(value.shape) != shape:
                raise ValueError(f"{key} 形状 {tuple(value.shape)} != {shape}")
        elif kind == "bool":
            if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
                raise ValueError(
                    f"{key} 必须是 bool torch.Tensor，got "
                    f"{getattr(value, 'dtype', type(value).__name__)}"
                )
            if tuple(value.shape) != shape:
                raise ValueError(f"{key} 形状 {tuple(value.shape)} != {shape}")
        elif kind == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{key} 必须是 int，got {type(value).__name__}")
        elif kind == "str":
            if not isinstance(value, str):
                raise ValueError(f"{key} 必须是 str，got {type(value).__name__}")
    return record


# --------------------------------------------------------------------------
# split / 索引 / 池化
# --------------------------------------------------------------------------

def wam_split_from_episode(episode_ids, task_ids):
    """episode_id % 10 → 0..7 train、8 val、9 test，返回 3 个 bool mask [N]。

    规则只依赖 episode_id，同一条 episode 的所有 anchor 必然同 split
    （禁止随机窗口切分）。task_ids 仅用于形状校验，签名保留供未来
    task 条件分层使用。
    """
    ep = torch.as_tensor(episode_ids, dtype=torch.long)
    task = torch.as_tensor(task_ids, dtype=torch.long)
    if task.shape != ep.shape:
        raise ValueError(
            f"episode_ids 与 task_ids 形状不一致: {tuple(ep.shape)} vs {tuple(task.shape)}"
        )
    r = ep % 10
    return r < 8, r == 8, r == 9


def wam_anchor_index(seq_len: int = SEQUENCE_LENGTH) -> int:
    """4 决策上下文的最后一个决策点索引 = 3（seq_len-1）。"""
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    return seq_len - 1


def wam_future_frame(decision_frame, k) -> int:
    """未来帧换算：帧号 = decision_frame + k。

    k 单位 = 动作列/帧偏移（ACTION_AXIS_UNITS="frame_offset"）。依据
    build_longtraj_features.py 的 action→frame 映射（target_idx =
    s + t*CONTROL_STRIDE + h，见该文件 503-507）：锚点决策帧 d 的 action
    列 h 就是帧偏移，action[h] 对应帧 decision_frame + h。因此未来目标帧
    d+k（k∈(6,24,48)）直接相加，不再乘 CONTROL_STRIDE。
    """
    return int(decision_frame) + int(k)


def future_frame_in_bounds(decision_frame, k, episode_last_frame) -> bool:
    """未来帧 d+k 是否落在 episode 内：帧号 ≤ episode 末帧帧号。

    k 单位 = 动作列/帧偏移。episode_last_frame = 该 episode 最后一帧的
    帧号（per-task 文件帧数 − 1）。+48 终点 frame d+48 超出 48 个动作列，
    perturbed_future 的窗口检查够不到它，必须由本函数单独判定；M1 有
    raw episode 时用真实 last_frame 调用，M0 fake 模式无帧数信息时给
    合理默认（True）。
    """
    return wam_future_frame(decision_frame, k) <= int(episode_last_frame)


def wam_last_slice_pool(h11_dense: torch.Tensor) -> torch.Tensor:
    """H11 dense [B,1152,768] → [B,16,768]：最后时间片 4×4 空间块均值。

    依据（backbones.py:1056-1070 ``forward_hierarchical_dense``）：
    1152 = 2 时间片 × 24×24，扁平顺序 t→y→x。``h11_dense[:, 576:, :]``
    即第二个（最新）时间片，reshape (B,24,24,768) 后是 y 行 × x 列的
    空间网格，再按 6×6 块取均值得到 4×4=16 空间 token（t→y→x 中 y 为
    主序，与 _dense_coords 的坐标顺序一致）。用 reshape+view 分块 mean，
    不用池化层；也刻意不用 pool_mtvj_coarse_tokens——那是 16 个连续
    一维桶，不是空间 4×4。
    """
    if h11_dense.ndim != 3:
        raise ValueError(
            f"h11_dense 必须是 [B,1152,D]，got {h11_dense.ndim} 维"
        )
    if h11_dense.shape[1] != H11_DENSE_TOKENS:
        raise ValueError(
            f"h11_dense 第二维必须是 {H11_DENSE_TOKENS}，got {h11_dense.shape[1]}"
        )
    dim = h11_dense.shape[-1]
    # 最后时间片 [B,576,D] → [B,24,24,D]（y 行 × x 列）
    last = h11_dense[:, LAST_SLICE_TOKENS:, :].contiguous()
    # [B, (4×6), (4×6), D] → 6×6 块均值 → [B,4,4,D] → [B,16,D]
    grid = last.view(
        last.shape[0], POOL_GRID, POOL_BLOCK, POOL_GRID, POOL_BLOCK, dim
    )
    return grid.mean(dim=(2, 4)).reshape(h11_dense.shape[0], N_SCENE_TOKENS, dim)


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

@dataclass
class WAMCacheManifest:
    """缓存 sidecar 契约（评测与训练不得自行猜测默认值）。"""
    contract: str = CONTRACT
    contract_version: int = CONTRACT_VERSION
    action_axis_units: str = ACTION_AXIS_UNITS
    base_ckpt_sha256: str = ""
    data_sha256: str = ""
    horizons: tuple = HORIZONS
    per_task_files: list = field(default_factory=list)  # index by task_id，空任务为 None
    n_anchors: int = 0
    normalization: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "contract": self.contract,
            "contract_version": self.contract_version,
            "action_axis_units": self.action_axis_units,
            "base_ckpt_sha256": self.base_ckpt_sha256,
            "data_sha256": self.data_sha256,
            "horizons": list(self.horizons),
            "per_task_files": self.per_task_files,
            "n_anchors": self.n_anchors,
            "normalization": self.normalization,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WAMCacheManifest":
        return cls(
            contract=d["contract"],
            contract_version=int(d["contract_version"]),
            action_axis_units=d["action_axis_units"],
            base_ckpt_sha256=d["base_ckpt_sha256"],
            data_sha256=d["data_sha256"],
            horizons=tuple(d["horizons"]),
            per_task_files=list(d["per_task_files"]),
            n_anchors=int(d["n_anchors"]),
            normalization=dict(d["normalization"]),
        )

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "WAMCacheManifest":
        return cls.from_dict(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------
# 记录填充（M1 GPU 管线）
# --------------------------------------------------------------------------

def _expect_tensor(x, shape: tuple, name: str) -> None:
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"{name} 必须是 torch.Tensor，got {type(x).__name__}")
    if tuple(x.shape) != tuple(shape):
        raise ValueError(f"{name} 形状 {tuple(x.shape)} != {tuple(shape)}")


def fill_record_latents(record: dict, base_outputs: dict) -> dict:
    """把冻结基座前向输出写入记录，替换 M0 占位零张量（M1 GPU 管线调用）。

    base_outputs 键（全部相对当前 anchor 决策点 d）：
      action_condition [48,512]          VA action condition
      va_layers        8×[16,512]        VA 每层记忆快照
      spatial16        [16,768]          wam_last_slice_pool(H11 dense) = z(d)
      geo8             [8]               g(d) = (p * visibility.unsqueeze(-1)).reshape(-1,8)
      future_latent    3×[16,768]        未来绝对 latent z(d+k)，顺序 = HORIZONS
      future_geo       3×[2,8]           (g_future, nu)，nu = g_future - g_current
    future_latent/future_geo 接受 list/tuple 或已 stack 的 [3,...] 张量；
    future_geo 额外接受 3 个 (g_future, nu) 对的 list（每对 [8]）。

    记录目标语义（trainer 直接取用，不再减当前值）：
      future_latent_target = z(d+k) − z(d)   （Δlatent，本函数内计算）
      future_geo_target    = (g_future, ν)   （原样存储，slice0 绝对、slice1 差分）
    就地修改并返回 record，便于链式 torch.save。形状不符即 ValueError。
    """
    required = (
        "action_condition", "va_layers", "spatial16", "geo8",
        "future_latent", "future_geo",
    )
    missing = [k for k in required if k not in base_outputs]
    if missing:
        raise ValueError(f"base_outputs 缺字段: {missing}")

    ac = base_outputs["action_condition"]
    _expect_tensor(ac, (ACTION_HORIZON, VA_DIM), "action_condition")

    va = list(base_outputs["va_layers"])
    if len(va) != N_VA_LAYERS:
        raise ValueError(f"va_layers 必须是 {N_VA_LAYERS} 层，got {len(va)}")
    for idx, layer in enumerate(va):
        _expect_tensor(layer, (VA_TOKENS, VA_DIM), f"va_layers[{idx}]")

    sp = base_outputs["spatial16"]
    _expect_tensor(sp, (N_SCENE_TOKENS, VISION_DIM), "spatial16")

    geo = base_outputs["geo8"]
    _expect_tensor(geo, (GEO_DIM,), "geo8")

    def _stack3(x, shape, name):
        if isinstance(x, torch.Tensor):
            _expect_tensor(x, (3, *shape), name)
            return x
        stacked = torch.stack([torch.as_tensor(t) for t in x])
        _expect_tensor(stacked, (3, *shape), name)
        return stacked

    fl = _stack3(base_outputs["future_latent"], (N_SCENE_TOKENS, VISION_DIM),
                 "future_latent")   # 绝对 z(d+k)
    if isinstance(base_outputs["future_geo"], torch.Tensor):
        fg = _stack3(base_outputs["future_geo"], (2, GEO_DIM), "future_geo")
    else:
        # list of 3 个 (g_future, nu) 对 → [3,2,8]
        fg = torch.stack([
            torch.stack([torch.as_tensor(t) for t in pair])
            for pair in base_outputs["future_geo"]
        ])
        _expect_tensor(fg, (3, 2, GEO_DIM), "future_geo")

    record["action_condition"] = ac.float()
    record["va_layers"] = [layer.float() for layer in va]
    record["spatial16"] = sp.float()
    record["geo8"] = geo.float()
    # Δlatent = z(d+k) − z(d)：trainer 不再减当前（cache 已存差分）。
    record["future_latent_target"] = (fl - sp).float()
    # (g_future, ν) 原样存储：slice0 = g(d+k) 绝对，slice1 = ν = g_future − g_current。
    record["future_geo_target"] = fg.float()
    return record


# --------------------------------------------------------------------------
# 构建
# --------------------------------------------------------------------------

def _sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _jsonable(x):
    if isinstance(x, torch.Tensor):
        return x.tolist() if x.ndim > 0 else x.item()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def _future_window_cols(k: int) -> list[int]:
    """未来窗口 [d+k-6, d+k-4, d+k-2, d+k] 的 action 列偏移。

    裁剪到 0..ACTION_HORIZON-1；k=48 时终点列 48（frame d+48）不在 48
    动作列内、且其有效性不在 action_valid_mask 语义内，因此窗口检查
    只覆盖 d+k-6..d+k-2 三列，+48 终点越界由 valid_future（
    future_frame_in_bounds）单独判定。
    """
    cols = [k - 6, k - 4, k - 2, k]
    return [c for c in cols if 0 <= c < ACTION_HORIZON]


def _build_anchor(payload: dict, i: int, *, anchor_row: int,
                  frame_counts: dict | None) -> dict:
    """第 i 个窗口的 anchor 记录（latent 字段为占位零张量，M1 填）。"""
    actions = payload["actions"][i, anchor_row]                 # [48,4]
    valid = payload["action_valid_mask"][i, anchor_row]         # [48] bool
    task_file, ep_idx, fidx = payload["frame_refs"][i]
    fidx = np.asarray(fidx)
    if fidx.shape != (SEQUENCE_LENGTH, SEQUENCE_LENGTH):
        raise ValueError(
            f"frame_refs[{i}] 帧窗形状 {fidx.shape} != "
            f"({SEQUENCE_LENGTH},{SEQUENCE_LENGTH})"
        )
    # 帧窗最后一列 = 决策点帧本身（clip_frame_indices 的历史帧窗以 d 结尾）；
    # anchor 决策点 d = s+18 >= 18，起点 clamp 不会触发。
    decision_frame = int(fidx[anchor_row, -1])

    perturbed = torch.zeros(len(HORIZONS), dtype=torch.bool)
    valid_future = torch.ones(len(HORIZONS), dtype=torch.bool)
    for h, k in enumerate(HORIZONS):
        # perturbed_future：只信 action_valid_mask（其已含成功后/settle/
        # 无效动作与 unseen-recovery 扰动跨越，见模块 docstring）。
        cols = _future_window_cols(k)
        if cols and bool((~valid[cols]).any().item()):
            perturbed[h] = True
        # valid_future：frame d+k 是否在 episode 内（+48 终点单独判定）。
        if frame_counts is not None:
            total = frame_counts.get(int(ep_idx))
            if total is None:
                raise ValueError(
                    f"cache 时间对齐失败：{task_file}:episode[{ep_idx}] "
                    f"在 per-task 文件中不存在"
                )
            valid_future[h] = future_frame_in_bounds(decision_frame, k, total - 1)

    return {
        "action_condition": torch.zeros(ACTION_HORIZON, VA_DIM),
        "va_layers": [torch.zeros(VA_TOKENS, VA_DIM) for _ in range(N_VA_LAYERS)],
        "spatial16": torch.zeros(N_SCENE_TOKENS, VISION_DIM),
        "geo8": torch.zeros(GEO_DIM),
        "actions": actions.clone(),
        "future_latent_target": torch.zeros(3, N_SCENE_TOKENS, VISION_DIM),
        "future_geo_target": torch.zeros(3, 2, GEO_DIM),
        "action_valid": valid.clone(),
        "perturbed_future": perturbed,
        "valid_future": valid_future,
        "horizon_weight": torch.tensor(HORIZON_WEIGHTS, dtype=torch.float32),
        "episode_id": int(payload["episode_id"][i]),
        "task_id": int(payload["instruction_id"][i]),
        "task_file": str(task_file),
        "ep_idx": int(ep_idx),
        "decision_frame": decision_frame,
    }


def _episode_frame_counts(task_file: str) -> dict[int, int] | None:
    """从 per-task longtraj 文件读每 episode 帧数（未来帧越界检测用）。

    文件缺失时返回 None（跳过越界检测），由 M1 填充时自查。
    """
    path = ROOT / "data" / f"metaworld_longtraj_{task_file}.pt"
    if not path.is_file():
        warnings.warn(
            f"wam cache: {path} 缺失，跳过该任务未来帧越界检测"
            f"（M1 填充 latent 时须自查 d+48 越界）", RuntimeWarning, stacklevel=2,
        )
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    counts = {ei: len(ep["frames"]) for ei, ep in enumerate(data["episodes"])}
    del data
    return counts


def _fake_payload(n_tasks: int = 3, eps_per_task: int = 10,
                  wins_per_ep: int = 2, seed: int = 0) -> dict:
    """--fake 模式的合成 windows payload（结构与真实文件一致）。

    30 个 episode（id 0..29，% 10 覆盖全部 split），每 ep 2 个窗口共 60
    anchor；ep 4 带扰动（recovery 跨 k=24/48 未来窗口）、ep 7 提前成功
    （post-success 无效帧），用于端到端验证 mask 逻辑。
    """
    g = torch.Generator().manual_seed(seed)
    n = n_tasks * eps_per_task * wins_per_ep
    actions = torch.zeros(n, SEQUENCE_LENGTH, ACTION_HORIZON, ACTION_DIM)
    valid = torch.ones(n, SEQUENCE_LENGTH, ACTION_HORIZON, dtype=torch.bool)
    recovery = torch.zeros(n, SEQUENCE_LENGTH, ACTION_HORIZON, dtype=torch.bool)
    ep_ids = torch.zeros(n, dtype=torch.long)
    task_ids = torch.zeros(n, dtype=torch.long)
    refs = []
    i = 0
    for t in range(n_tasks):
        tf = f"fake-task-{t:02d}"
        for e in range(eps_per_task):
            ep_id = t * eps_per_task + e
            for w in range(wins_per_ep):
                actions[i] = (torch.randn(
                    SEQUENCE_LENGTH, ACTION_HORIZON, ACTION_DIM, generator=g
                ) * 0.3).clamp(-1, 1)
                if e == 4:  # 扰动 episode：模拟 unseen-recovery 跨越——
                    # 真实文件里 valid 已被 phase1 减掉不可观测 recovery
                    # 目标，此处照抄该语义（recovery 置位 + valid 清除）。
                    recovery[i, 3, 24:ACTION_HORIZON] = True
                    valid[i, 3, 24:ACTION_HORIZON] = False
                if e == 7:  # 提前成功 episode：30 步后无效
                    valid[i, 3, 30:] = False
                ep_ids[i] = ep_id
                task_ids[i] = t
                start = w * CONTROL_STRIDE
                rows = []
                for tt in range(SEQUENCE_LENGTH):
                    d_t = start + tt * CONTROL_STRIDE
                    rows.append([max(d_t - off, 0) for off in (6, 4, 2, 0)])
                refs.append((tf, e, rows))
                i += 1
    assert i == n
    return {
        "actions": actions,
        "action_valid_mask": valid,
        "recovery_mask": recovery,
        "episode_id": ep_ids,
        "instruction_id": task_ids,
        "frame_refs": refs,
        "normalization": {
            "action_q01": torch.tensor(-1.0), "action_q99": torch.tensor(1.0),
            "state_q01": torch.tensor(-1.0), "state_q99": torch.tensor(1.0),
        },
        "metadata": {
            "contract": "fake", "fps": 80, "control_stride": CONTROL_STRIDE,
            "action_horizon": ACTION_HORIZON,
            "tasks": [f"fake-task-{t}" for t in range(n_tasks)],
        },
    }


def _fake_latent_fn():
    """fake 模式按 anchor 索引确定性生成合成 base_outputs 并填充记录。

    语义与真实 M1 管线一致：future_latent 传绝对 z(d+k)，由
    fill_record_latents 转为 Δlatent = z(d+k) − z(d)；future_geo 传
    (g_future, ν = g_future − g_current) 对，原样入记录。
    """

    def fill(record: dict, i: int) -> None:
        g = torch.Generator().manual_seed(1000 + i)
        sp = torch.randn(N_SCENE_TOKENS, VISION_DIM, generator=g)      # z(d)
        geo = torch.randn(GEO_DIM, generator=g)                         # g(d)
        z_future = [torch.randn(N_SCENE_TOKENS, VISION_DIM, generator=g)
                    for _ in range(3)]                                   # z(d+k) 绝对
        g_future = [torch.randn(GEO_DIM, generator=g) for _ in range(3)]  # g(d+k) 绝对
        future_geo = [
            (gf, gf - geo) for gf in g_future
        ]   # ν = g_future − g_current（与 M1 语义一致）
        fill_record_latents(record, {
            "action_condition": torch.randn(ACTION_HORIZON, VA_DIM, generator=g),
            "va_layers": [torch.randn(VA_TOKENS, VA_DIM, generator=g)
                          for _ in range(N_VA_LAYERS)],
            "spatial16": sp,
            "geo8": geo,
            "future_latent": z_future,
            "future_geo": future_geo,
        })

    return fill


def build_wam_cache(windows_pt, out_dir, *, base_ckpt, device: str = "cpu",
                    max_tasks: int = 49) -> WAMCacheManifest:
    """流式读 windows 文件 → 每 anchor 一条记录 → 按 task 分片 torch.save。

    windows_pt=None 即 --fake 合成模式（合成小数据集走完整管线，含合成
    latent）。真实模式下 latent 字段为零占位（M1 GPU 管线经
    fill_record_latents 填充后重写分片）。device 保留给 M1 GPU 编码。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if max_tasks < 1:
        raise ValueError(f"max_tasks must be >= 1, got {max_tasks}")

    if windows_pt is None:
        payload = _fake_payload()
        data_sha = ""
    else:
        windows_pt = Path(windows_pt)
        if not windows_pt.is_file():
            raise FileNotFoundError(f"windows 文件不存在: {windows_pt}")
        payload = torch.load(windows_pt, map_location="cpu", weights_only=False)
        data_sha = _sha256_file(windows_pt)

    # ---- 输入契约校验（只接受 h48 windows 文件） ----
    actions_t = payload["actions"]
    if (actions_t.ndim != 4 or actions_t.shape[1] != SEQUENCE_LENGTH
            or actions_t.shape[2:] != (ACTION_HORIZON, ACTION_DIM)):
        raise ValueError(
            f"actions 必须是 [N,{SEQUENCE_LENGTH},{ACTION_HORIZON},"
            f"{ACTION_DIM}]，got {tuple(actions_t.shape)}；wam cache 只接受 h48 windows"
        )
    n = int(actions_t.shape[0])
    for key, shape in (("action_valid_mask", (n, SEQUENCE_LENGTH, ACTION_HORIZON)),
                       ("recovery_mask", (n, SEQUENCE_LENGTH, ACTION_HORIZON)),
                       ("episode_id", (n,)), ("instruction_id", (n,))):
        tensor = payload.get(key)
        if tensor is None or tuple(tensor.shape) != shape:
            raise ValueError(f"{key} 形状不符：期望 {shape}，got "
                             f"{getattr(tensor, 'shape', None)}")
    refs = payload["frame_refs"]
    if len(refs) != n:
        raise ValueError(f"frame_refs 长度 {len(refs)} != 样本数 {n}")

    base_sha = _sha256_file(Path(base_ckpt)) if base_ckpt else ""

    task_ids_np = payload["instruction_id"].numpy()
    tids = sorted({int(t) for t in task_ids_np})[:max_tasks]
    if not tids:
        raise ValueError("windows 文件不含任何任务")
    anchor_row = wam_anchor_index(SEQUENCE_LENGTH)
    latent_fn = _fake_latent_fn() if windows_pt is None else None

    per_task_files: list = [None] * (max(tids) + 1)
    index_shards = []
    n_anchors = 0
    for tid in tids:
        idxs = np.flatnonzero(task_ids_np == tid)
        task_file = refs[int(idxs[0])][0]
        if any(refs[int(j)][0] != task_file for j in idxs):
            raise ValueError(f"task_id={tid} 的窗口跨多个 task_file，数据不一致")
        frame_counts = None
        if windows_pt is not None:
            frame_counts = _episode_frame_counts(task_file)
        records, ep_list, t_list = [], [], []
        for j in idxs:
            j = int(j)
            record = _build_anchor(payload, j, anchor_row=anchor_row,
                                   frame_counts=frame_counts)
            if latent_fn is not None:
                latent_fn(record, j)
            assert_record_schema(record)  # 写分片前白名单校验
            records.append(record)
            ep_list.append(record["episode_id"])
            t_list.append(record["task_id"])
        shard_name = f"task_{tid:02d}.pt"
        torch.save({"records": records}, out_dir / shard_name)
        per_task_files[tid] = shard_name
        index_shards.append({
            "file": shard_name, "episode_id": ep_list, "task_id": t_list,
        })
        n_anchors += len(records)

    manifest = WAMCacheManifest(
        contract=CONTRACT,
        contract_version=CONTRACT_VERSION,
        action_axis_units=ACTION_AXIS_UNITS,
        base_ckpt_sha256=base_sha,
        data_sha256=data_sha,
        horizons=HORIZONS,
        per_task_files=per_task_files,
        n_anchors=n_anchors,
        normalization=_jsonable(payload.get("normalization", {})),
    )
    manifest.save(out_dir / "manifest.json")
    with open(out_dir / "index.json", "w") as f:
        json.dump({"shards": index_shards}, f, indent=2)
    return manifest


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class WAMCacheDataset(Dataset):
    """按 split mask 读取分片记录；__getitem__ 返回 record dict。

    构造时仅加载 index.json（每分片 episode_id/task_id 小索引），按
    wam_split_from_episode 计算全局 (分片, 局部) 行号表；分片记录按需
    torch.load 并缓存最近 2 个分片。episode 整条归属保证同一 episode
    的所有 anchor 落在同一 split。
    """

    def __init__(self, out_dir, manifest: WAMCacheManifest | None = None,
                 split: str = "train"):
        if split not in SPLITS:
            raise ValueError(f"split 必须是 {SPLITS} 之一，got {split!r}")
        self.out_dir = Path(out_dir)
        if manifest is None:
            manifest = self.out_dir / "manifest.json"
        if isinstance(manifest, (str, Path)):
            manifest = WAMCacheManifest.load(manifest)
        if not isinstance(manifest, WAMCacheManifest):
            raise TypeError(
                f"manifest 必须是 WAMCacheManifest 或路径，got {type(manifest).__name__}"
            )
        if manifest.contract != CONTRACT:
            raise ValueError(
                f"manifest contract {manifest.contract!r} != {CONTRACT!r}"
                f"（sidecar 不匹配，训练/评测不得自行猜测默认值）"
            )
        if int(manifest.contract_version) != CONTRACT_VERSION:
            raise ValueError(
                f"manifest contract_version {manifest.contract_version} != "
                f"{CONTRACT_VERSION}（cache/trainer schema 不匹配，须重建 cache）"
            )
        self.manifest = manifest
        index_path = self.out_dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"{index_path} 缺失（先运行 build_wam_cache 生成缓存）"
            )
        index = json.loads(index_path.read_text())
        rows: list[tuple[int, int]] = []
        for shard_i, sh in enumerate(index["shards"]):
            ep = torch.as_tensor(sh["episode_id"], dtype=torch.long)
            task = torch.as_tensor(sh["task_id"], dtype=torch.long)
            masks = wam_split_from_episode(ep, task)
            sel = {"train": masks[0], "val": masks[1], "test": masks[2]}[split]
            for local in sel.nonzero(as_tuple=False).flatten().tolist():
                rows.append((shard_i, int(local)))
        self._rows = rows
        self._shard_files = [sh["file"] for sh in index["shards"]]
        self._records_cache: dict[int, list] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index) -> dict:
        shard_i, local = self._rows[index]
        records = self._records_cache.get(shard_i)
        if records is None:
            if len(self._records_cache) >= 2:
                self._records_cache.clear()
            shard = torch.load(
                self.out_dir / self._shard_files[shard_i],
                map_location="cpu", weights_only=False,
            )
            records = shard["records"]
            self._records_cache[shard_i] = records
        return records[local]


if __name__ == "__main__":
    # 计划 Step 1 demo：合成 episode id 0..29 × 3 task，split 无泄漏且 8:1:1
    ep = torch.arange(30)          # 30 个唯一 episode id
    task = torch.arange(30) % 3
    tr, va, te = wam_split_from_episode(ep, task)
    assert (tr | va | te).all() and not (tr & va).any() \
        and not (va & te).any() and not (tr & te).any(), "split 泄漏"
    assert int(tr.sum()) == 24 and int(va.sum()) == 3 and int(te.sum()) == 3, \
        "split 比例不是 8:1:1"
    assert wam_anchor_index() == 3
    # 单测：k 单位 = 动作列/帧偏移（帧号 = decision_frame + k）
    assert wam_future_frame(100, 24) == 124
    assert wam_future_frame(0, 48) == 48
    # 单测：future_frame_in_bounds（+48 终点越界判定）
    assert future_frame_in_bounds(100, 48, 147) is False   # 148 > 147
    assert future_frame_in_bounds(100, 48, 148) is True    # 148 <= 148
    assert future_frame_in_bounds(0, 6, 5) is False        # 6 > 5
    assert future_frame_in_bounds(0, 6, 6) is True
    z = wam_last_slice_pool(torch.randn(2, 1152, 768))
    assert z.shape == (2, 16, 768)
    man = build_wam_cache(None, "/tmp/wam_cache_demo", base_ckpt=None)
    assert man.contract_version == CONTRACT_VERSION
    assert man.action_axis_units == ACTION_AXIS_UNITS
    ds = WAMCacheDataset("/tmp/wam_cache_demo", man, split="train")
    r = ds[0]
    assert r["actions"].shape == (48, 4)
    assert r["future_latent_target"].shape == (3, 16, 768)
    assert_record_schema(r)  # 读回记录过白名单
    # 单测：fake 记录目标语义（Δlatent / (g_future, ν)）与 schema 白名单
    bad = dict(r)
    del bad["valid_future"]
    try:
        assert_record_schema(bad)
        raise AssertionError("缺键记录应被白名单拒绝")
    except ValueError:
        pass
    print(f"demo ok: splits {int(tr.sum())}/{int(va.sum())}/{int(te.sum())}, "
          f"anchors={man.n_anchors}, train rows={len(ds)}")
