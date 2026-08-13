#!/usr/bin/env python
"""build_all49_clean_merge.py — generic N-task clean-resample merge into all49 windows.

Contract (mirrors build_hpfix_merge.py precedent):
- base: data/metaworld_longtraj_windows_h48_all49_repaired_v2_hpfix.pt
  (24014 windows; handle-pull already replaced; pair_id max 24603;
   episode blocks ...480023 + hp block 260000..260029)
- for each task in --tasks:
    * tid = base metadata.tasks.index(ENV_TO_TASK[task])
    * remove ALL base rows with instruction_id == tid
    * load {window-dir}/windows_{task}_fixed.pt, append its rows at the end
      (task order = --tasks order, same append-at-end pattern as hpfix)
    * pair_id: remapped to consecutive ids starting at current global max + 1
    * episode_id: remapped to its own 1000-aligned block starting at
      align1000(current_global_max_episode_id + 1000) + k*1000, local
      episodes 0..29 mapped identically (preserves episode % 10 train/val/test
      structure; block % 10 == 0)
    * metadata.source_files[src_idx] -> data/metaworld_longtraj_{task}_fixed.pt
      (src_idx found by basename match, NOT tid — source_files order differs
      from tasks order in the base)
    * metadata.clean_resample = list of merged tasks; n_trajectories recomputed
- key set + key order must equal base exactly; normalization/tasks/contract of
  each fixed windows file validated against base before merging.

Safety:
- NEVER overwrites any existing .pt: refuses if --output exists; writes
  {output}.tmp then os.replace. Window/traj files are only read.
- CPU only: forces CUDA_VISIBLE_DEVICES="" before torch import.
- Post-merge validation runs automatically; on any failure the output file is
  deleted.

Usage:
  # dry-run: full inspection, no write
  python scripts/build_all49_clean_merge.py --dry-run
  # real merge (only when all 11 windows_*_fixed.pt are ready)
  python scripts/build_all49_clean_merge.py
  # single-task trial merge into /tmp (logic check while resample still runs)
  python scripts/build_all49_clean_merge.py --tasks coffee-button-v3 \
      --output /tmp/test_clean_merge_coffee.pt
"""
import argparse
import io
import os
import sys

# 禁止 GPU：必须在导入 torch 之前强制清空可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402
import numpy as np  # noqa: E402

ROOT = "/home/ryan/Documents/robot/ORA0"
DATA = f"{ROOT}/data"
DEFAULT_BASE = f"{DATA}/metaworld_longtraj_windows_h48_all49_repaired_v2_hpfix.pt"
DEFAULT_OUT = f"{DATA}/metaworld_longtraj_windows_h48_all49_repaired_v2_clean.pt"
DEFAULT_TASKS = [
    # 多 seed 核验后最终合并的 11 项（其余 4 项：stick-pull、plate-slide-back-side、
    # handle-press、handle-press-side 核验无污染/重采反而降覆盖，不合并）
    "coffee-button-v3",
    "door-lock-v3",
    "door-unlock-v3",
    "faucet-open-v3",
    "faucet-close-v3",
    "handle-pull-side-v3",
    "lever-pull-v3",
    "window-open-v3",
    "window-close-v3",
    "plate-slide-v3",
    "plate-slide-side-v3",
]

# 与 scripts/build_longtraj_features.py 的 ENV_TO_TASK 完全一致（49 项）
ENV_TO_TASK = {
    "assembly-v3": "Pick up a nut and place it onto a peg",
    "basketball-v3": "Dunk the basketball into the basket",
    "bin-picking-v3": "Grasp the puck from one bin and place it into another bin",
    "box-close-v3": "Grasp the cover and close the box with it",
    "button-press-topdown-v3": "Press a button from the top",
    "button-press-topdown-wall-v3": "Bypass a wall and press a button from the top",
    "button-press-v3": "Press a button",
    "button-press-wall-v3": "Bypass a wall and press a button",
    "coffee-button-v3": "Push a button on the coffee machine",
    "coffee-pull-v3": "Pull a mug from a coffee machine",
    "coffee-push-v3": "Push a mug under a coffee machine",
    "dial-turn-v3": "Rotate a dial 180 degrees",
    "disassemble-v3": "Pick a nut out of a peg",
    "door-close-v3": "Close a door with a revolving joint",
    "door-lock-v3": "Lock the door by rotating the lock clockwise",
    "door-open-v3": "Open a door with a revolving joint",
    "door-unlock-v3": "Unlock the door by rotating the lock counter-clockwise",
    "hand-insert-v3": "Insert the gripper into a hole",
    "drawer-close-v3": "Push and close a drawer",
    "drawer-open-v3": "Open a drawer",
    "faucet-open-v3": "Rotate the faucet counter-clockwise",
    "faucet-close-v3": "Rotate the faucet clockwise",
    "hammer-v3": "Hammer a screw on the wall",
    "handle-press-side-v3": "Press a handle down sideways",
    "handle-press-v3": "Press a handle down",
    "handle-pull-side-v3": "Pull a handle up sideways",
    "handle-pull-v3": "Pull a handle up",
    "lever-pull-v3": "Pull a lever down 90 degrees",
    "pick-place-wall-v3": "Pick a puck, bypass a wall and place the puck",
    "pick-out-of-hole-v3": "Pick up a puck from a hole",
    "pick-place-v3": "Pick and place a puck to a goal",
    "plate-slide-v3": "Slide a plate into a cabinet",
    "plate-slide-side-v3": "Slide a plate into a cabinet sideways",
    "plate-slide-back-v3": "Get a plate from the cabinet",
    "plate-slide-back-side-v3": "Get a plate from the cabinet sideways",
    "peg-insert-side-v3": "Insert a peg sideways",
    "peg-unplug-side-v3": "Unplug a peg sideways",
    "soccer-v3": "Kick a soccer into the goal",
    "stick-push-v3": "Grasp a stick and push a box using the stick",
    "stick-pull-v3": "Grasp a stick and pull a box with the stick",
    "push-v3": "Push the puck to a goal",
    "push-wall-v3": "Bypass a wall and push a puck to a goal",
    "reach-v3": "Reach a goal position",
    "reach-wall-v3": "Bypass a wall and reach a goal",
    "shelf-place-v3": "Pick and place a puck onto a shelf",
    "sweep-into-v3": "Sweep a puck into a hole",
    "sweep-v3": "Sweep a puck off the table",
    "window-open-v3": "Push and open a window",
    "window-close-v3": "Push and close a window",
}


def log(msg):
    print(msg, flush=True)


def add_numpy_safe_globals():
    """轨迹文件含 numpy 数组/标量：白名单后仍保持 weights_only 安全语义."""
    import numpy.dtypes as _dt
    torch.serialization.add_safe_globals([
        np.core.multiarray._reconstruct, np.core.multiarray.scalar,
        np.dtype, np.ndarray,
    ] + [getattr(_dt, n) for n in dir(_dt) if n.endswith("DType")])


def find_tid(tasks_list, env):
    text = ENV_TO_TASK[env]
    try:
        return tasks_list.index(text)
    except ValueError as exc:
        raise ValueError(f"{env}: task text {text!r} absent from base tasks") from exc


def find_src_idx(base_meta, env):
    """source_files 顺序 ≠ tasks 顺序（如 faucet-close 是 src[19] 但 tid=21），
    必须按 basename 精确匹配."""
    target = f"metaworld_longtraj_{env}.pt"
    hits = [i for i, s in enumerate(base_meta["source_files"])
            if os.path.basename(s) == target]
    if len(hits) != 1:
        raise ValueError(f"{env}: expected exactly 1 source_files entry {target}, got {hits}")
    return hits[0]


def align1000(x):
    return ((x + 999) // 1000) * 1000


def check_fixed_structure(fx, base, env, tid):
    """fixed windows 文件与 base 的键/契约一致性（逐项断言，失败即抛）."""
    assert set(fx.keys()) == set(base.keys()), f"{env}: key set differs"
    assert list(fx.keys()) == list(base.keys()), f"{env}: key order differs"
    n_new = fx["actions"].shape[0]
    assert n_new > 0, f"{env}: fixed windows empty"
    assert (fx["instruction_id"] == tid).all(), f"{env}: instruction_id not all {tid}"
    task_files = {r[0] for r in fx["frame_refs"]}
    assert task_files == {f"{env}_fixed"}, f"{env}: bad frame_refs task_file {task_files}"
    assert all(len(r) == 3 for r in fx["frame_refs"]), f"{env}: frame_refs len != 3"
    # episode: 本地 0..29（单输入文件 fi=0 → ep_id == ei），block 对齐后保持 %10
    loc_eps = torch.unique(fx["episode_id"])
    assert int(loc_eps.max()) <= 29 and int(loc_eps.min()) >= 0, \
        f"{env}: local episode ids out of 0..29: {loc_eps.tolist()}"
    # metadata / normalization 与 base 一致
    fm = fx["metadata"]
    bm = base["metadata"]
    assert fm["tasks"] == bm["tasks"], f"{env}: metadata.tasks differs"
    assert fm["contract"] == bm["contract"], f"{env}: contract differs"
    assert fm["contract_version"] == bm["contract_version"], f"{env}: contract_version differs"
    for k in ("fps", "control_stride", "action_horizon", "action_contract"):
        assert fm.get(k) == bm.get(k), f"{env}: metadata.{k} differs: {fm.get(k)!r} vs {bm.get(k)!r}"
    for k in base["normalization"]:
        assert torch.equal(fx["normalization"][k], base["normalization"][k]), \
            f"{env}: normalization.{k} differs"
    srcs = fm.get("source_files", [])
    assert len(srcs) == 1, f"{env}: fixed source_files len {len(srcs)} != 1"
    assert os.path.basename(srcs[0]) == f"metaworld_longtraj_{env}_fixed.pt", \
        f"{env}: unexpected fixed traj path {srcs[0]}"
    return n_new, int(len(loc_eps))


def build_plan(base, tasks, window_dir):
    n_base = base["actions"].shape[0]
    tasks_list = base["metadata"]["tasks"]
    plan = []
    for env in tasks:
        tid = find_tid(tasks_list, env)
        src_idx = find_src_idx(base["metadata"], env)
        old_mask = base["instruction_id"] == tid
        n_old = int(old_mask.sum().item())
        old_eps = int(len(torch.unique(base["episode_id"][old_mask])))
        wpath = os.path.join(window_dir, f"windows_{env}_fixed.pt")
        tpath = os.path.join(DATA, f"metaworld_longtraj_{env}_fixed.pt")
        wready = os.path.isfile(wpath)
        n_new = n_new_eps = None
        werr = None
        if wready:
            try:
                fx = torch.load(wpath, map_location="cpu", weights_only=True)
                n_new, n_new_eps = check_fixed_structure(fx, base, env, tid)
                del fx
            except Exception as exc:  # 文件存在但损坏/契约不符：按未就绪处理
                werr = f"{type(exc).__name__}: {exc}"
                wready = False
        plan.append(dict(env=env, tid=tid, src_idx=src_idx, n_old=n_old,
                         old_eps=old_eps, wpath=wpath, tpath=tpath,
                         wready=wready, n_new=n_new, new_eps=n_new_eps,
                         werr=werr))
    return plan, n_base


def print_dry_run(base, plan, n_base, base_path):
    log("=" * 90)
    log(f"DRY-RUN: base={base_path}")
    log(f"base rows={n_base}  pair_id max={int(base['pair_id'].max())}  "
        f"episode_id max={int(base['episode_id'].max())}")
    ep_start = align1000(int(base["episode_id"].max()) + 1000)
    pair_start = int(base["pair_id"].max()) + 1
    log(f"planned episode block start={ep_start} (align1000(global_max+1000))  "
        f"pair_id start={pair_start}")
    log("-" * 90)
    log(f"{'#':>2} {'task':26s} {'tid':>3} {'old_rows':>8} {'old_eps':>7} "
        f"{'window_ready':>12} {'new_rows':>8} {'new_eps':>7}  {'ep_block':>9}")
    sum_old = 0
    sum_new_known = 0
    missing = []
    for k, p in enumerate(plan):
        block = ep_start + k * 1000
        if p["n_new"] is None:
            new_s, new_e = "?", "?"
        else:
            new_s, new_e = str(p["n_new"]), str(p["new_eps"])
            sum_new_known += p["n_new"]
        log(f"{k:>2} {p['env']:26s} {p['tid']:>3} {p['n_old']:>8} "
            f"{p['old_eps']:>7} {'YES' if p['wready'] else 'NO':>12} "
            f"{new_s:>8} {new_e:>7}  {block}..{block+29}")
        sum_old += p["n_old"]
        if not p["wready"]:
            missing.append(p["env"])
    log("-" * 90)
    log(f"sum_old_rows={sum_old}  sum_new_rows(known)={sum_new_known}  "
        f"expected_total={n_base - sum_old + sum_new_known}"
        + (" (final total needs all files ready)" if missing else " (all ready)"))
    if not missing:
        log(f"n_trajectories: base {base['metadata']['n_trajectories']} -> "
            f"{base['metadata']['n_trajectories'] - sum(p['old_eps'] for p in plan) + sum(p['new_eps'] for p in plan)}")
    if missing:
        log("")
        log(f"MISSING {len(missing)}/{len(plan)} window files (resample still running or failed):")
        for p in plan:
            if not p["wready"]:
                env = p["env"]
                tag = "INVALID" if p["werr"] else "missing"
                log(f"  - {os.path.join('data', f'windows_{env}_fixed.pt')} [{tag}]"
                    + (f" {p['werr']}" if p["werr"] else ""))
        log("merge NOT executed (dry-run).")
        return 1
    log(f"all {len(plan)} window files ready; dry-run OK (no file written).")
    return 0


def frame_decode_check(merged, base, tasks, new_row_ids):
    """抽 2 个被替换行（第一个与最后一个新行）按 frame_refs 解码帧
    （与 validate_frames_hpfix.py 相同方式：轨迹文件含 JPEG bytes，
    需 weights_only=False + numpy 白名单）。"""
    from PIL import Image
    add_numpy_safe_globals()
    sel = [new_row_ids[0], new_row_ids[-1]]
    ok = True
    for i in sel:
        task_file, ep_idx, fidx = merged["frame_refs"][i]
        env = task_file[:-len("_fixed")]
        assert env in tasks, f"decode: unknown fixed task_file {task_file!r}"
        src_idx = find_src_idx(base["metadata"], env)
        traj_path = merged["metadata"]["source_files"][src_idx]
        assert os.path.isfile(traj_path), f"decode: missing traj {traj_path}"
        traj = torch.load(traj_path, map_location="cpu", weights_only=False)
        ep_frames = traj["episodes"][ep_idx]["frames"]
        nf = len(ep_frames)
        fidx_np = np.asarray(fidx)
        in_bounds = bool((fidx_np >= 0).all() and (fidx_np < nf).all())
        log(f"  decode row {i}: {task_file} ep {ep_idx} ({nf} frames), "
            f"fidx {fidx_np.shape} in-bounds={in_bounds}")
        ok &= in_bounds
        if not in_bounds:
            continue
        for tag, f in [("first", int(fidx_np[0, 0])), ("last", int(fidx_np[-1, -1]))]:
            img = np.asarray(Image.open(io.BytesIO(ep_frames[f])).convert("RGB"),
                             dtype=np.uint8)
            good = img.ndim == 3 and img.shape[2] == 3 and img.max() > 5
            log(f"    {tag} frame {f}: shape={img.shape} non-blank={good}")
            ok &= good
        del traj
    return ok


def validate_merged(merged, base, plan, tasks, out_path):
    """合并后全量校验；任一失败返回 False（调用方删除输出文件）。"""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        d = str(detail)
        if len(d) > 160:
            d = d[:157] + "..."
        log(f"[{status}] {name}" + (f"  ({d})" if detail else ""))

    n = merged["actions"].shape[0]
    n_base = base["actions"].shape[0]
    check("keys set identical", set(merged.keys()) == set(base.keys()),
          f"merged-only={set(merged)-set(base)} base-only={set(base)-set(merged)}")
    check("key order identical", list(merged.keys()) == list(base.keys()))
    for k, v in merged.items():
        if isinstance(v, torch.Tensor) and k not in ("normalization",):
            check(f"rows consistent: {k}", v.shape[0] == n, str(tuple(v.shape)))
    check("frame_refs len == rows", len(merged["frame_refs"]) == n)

    # (b) counts
    expected = n_base - sum(p["n_old"] for p in plan) + sum(p["n_new"] for p in plan)
    check(f"total rows == base - sum(old) + sum(new) = {expected}", n == expected, str(n))
    log("  per-task old -> new:")
    ep_start = align1000(int(base["episode_id"].max()) + 1000)
    kept = {}
    for k, p in enumerate(plan):
        kept[p["tid"]] = p["n_new"]
        log(f"    {p['env']:26s} tid={p['tid']:>2} {p['n_old']:>4} -> {p['n_new']:>4}"
            f"  (episode block {ep_start + k * 1000}..{ep_start + k * 1000 + 29})")

    # (c) replaced tasks' frame_refs point to _fixed; others identical to base
    a_i = base["instruction_id"].numpy()
    n_i = merged["instruction_id"].numpy()
    ca = {int(t): int((a_i == t).sum()) for t in np.unique(a_i)}
    cn = {int(t): int((n_i == t).sum()) for t in np.unique(n_i)}
    for p in plan:
        tid = p["tid"]
        refs = [merged["frame_refs"][i] for i in (n_i == tid).nonzero()[0]]
        files = {r[0] for r in refs}
        check(f"{p['env']}: frame_refs all _fixed", files == {f"{p['env']}_fixed"}, str(files))
    others = [t for t in ca if t not in kept]
    diffs = {t: (ca[t], cn[t]) for t in others if cn.get(t) != ca[t]}
    check(f"{len(others)} other tasks counts identical", len(diffs) == 0, str(diffs))
    check("instruction_id sets identical", set(ca.keys()) == set(cn.keys()))

    # (d) language_hidden bitwise identical to old rows per replaced task
    for p in plan:
        tid = p["tid"]
        old_lh = base["language_hidden"][base["instruction_id"] == tid]
        new_lh = merged["language_hidden"][merged["instruction_id"] == tid]
        old_uniform = bool((old_lh == old_lh[0]).all().item())
        same = bool((new_lh == old_lh[0]).all().item()) if old_uniform else False
        check(f"{p['env']}: language_hidden == old task rows (bitwise)",
              old_uniform and same)
        old_lm = base["language_mask"][base["instruction_id"] == tid]
        new_lm = merged["language_mask"][merged["instruction_id"] == tid]
        check(f"{p['env']}: language_mask == old task rows",
              bool((old_lm == old_lm[0]).all().item()) and
              bool((new_lm == old_lm[0]).all().item()))

    # (e) value ranges / NaN
    act = merged["actions"]
    in_range = bool(((act >= -1.0 - 1e-6) & (act <= 1.0 + 1e-6)).all().item())
    check("actions in [-1, 1]", in_range,
          f"min={act.min().item():.4f} max={act.max().item():.4f}")
    check("actions no NaN", not torch.isnan(act).any())
    check("previous_action/proprio no NaN",
          not torch.isnan(merged["previous_action"]).any() and
          not torch.isnan(merged["proprio"]).any())
    check("masks no NaN",
          not torch.isnan(merged["action_valid_mask"].float()).any() and
          not torch.isnan(merged["recovery_mask"].float()).any() and
          not torch.isnan(merged["decision_recovery"].float()).any() and
          not torch.isnan(merged["door_metric_state_valid"].float()).any() and
          not torch.isnan(merged["language_mask"].float()).any())
    check("action_valid_mask dtype bool", merged["action_valid_mask"].dtype == torch.bool)

    # (f) frame decode of 2 replaced rows
    new_rows = [i for i in range(n) if int(n_i[i]) in kept]
    try:
        dec_ok = frame_decode_check(merged, base, tasks, new_rows)
        check("frame decode: 2 rows", dec_ok)
    except Exception as exc:  # 解码异常视为校验失败（调用方删输出）
        check("frame decode: 2 rows", False, repr(exc))

    # (g) global uniqueness
    check("pair_id globally unique", len(torch.unique(merged["pair_id"])) == n,
          f"{len(torch.unique(merged['pair_id']))}/{n}")
    n_eps = len(torch.unique(merged["episode_id"]))
    expected_eps = int(len(torch.unique(base["episode_id"]))) \
        - sum(p["old_eps"] for p in plan) \
        + sum(p["new_eps"] for p in plan)
    check(f"episode_id globally unique ({n_eps} == {expected_eps})", n_eps == expected_eps)
    for k, p in enumerate(plan):
        tid = p["tid"]
        eps = merged["episode_id"][merged["instruction_id"] == tid]
        lo, hi = ep_start + k * 1000, ep_start + k * 1000 + 29
        check(f"{p['env']}: episodes within block {lo}..{hi}",
              bool((eps >= lo).all() and (eps <= hi).all()),
              f"{int(eps.min())}..{int(eps.max())}")

    # (h) metadata
    bm = base["metadata"]
    mm = merged["metadata"]
    for k in bm:
        if k in ("source_files", "n_trajectories"):
            continue
        check(f"metadata.{k} preserved", mm.get(k) == bm[k],
              f"{bm[k]!r} vs {mm.get(k)!r}")
    check("clean_resample recorded", mm.get("clean_resample") == tasks, str(mm.get("clean_resample")))
    check("n_source_files stays 49", mm["n_source_files"] == 49 and len(mm["source_files"]) == 49)
    for p in plan:
        f = mm["source_files"][p["src_idx"]]
        check(f"{p['env']}: source_files -> _fixed path",
              os.path.basename(f) == f"metaworld_longtraj_{p['env']}_fixed.pt"
              and os.path.isfile(f), f)
    for k in base["normalization"]:
        check(f"normalization.{k} == base",
              torch.equal(merged["normalization"][k], base["normalization"][k]))

    # (i) 磁盘重载：输出必须 weights_only=True 可加载且键与行数与内存一致
    try:
        chk = torch.load(out_path, map_location="cpu", weights_only=True)
        check("output reloads with weights_only=True", isinstance(chk, dict))
        check("reloaded key set == base", set(chk.keys()) == set(base.keys()))
        check("reloaded key order == base", list(chk.keys()) == list(base.keys()))
        check("reloaded rows == in-memory rows",
              chk["actions"].shape[0] == n and chk["pair_id"].max().item() == merged["pair_id"].max().item())
        del chk
    except Exception as exc:
        check("output reload (weights_only=True)", False, repr(exc))
    return ok


def merge(base, plan, tasks):
    n_base = base["actions"].shape[0]
    keep = torch.ones(n_base, dtype=torch.bool)
    ep_start = align1000(int(base["episode_id"].max()) + 1000)
    pair_cursor = int(base["pair_id"].max())
    extra = {k: [] for k in base if isinstance(base[k], torch.Tensor)}
    extra_refs = []  # frame_refs
    meta_src = list(base["metadata"]["source_files"])
    plan = [dict(p) for p in plan]
    for k, p in enumerate(plan):
        env, tid, src_idx = p["env"], p["tid"], p["src_idx"]
        keep &= base["instruction_id"] != tid
        fx = torch.load(p["wpath"], map_location="cpu", weights_only=True)
        n_new, n_new_eps = check_fixed_structure(fx, base, env, tid)
        p["n_new"], p["new_eps"] = n_new, n_new_eps
        log(f"merge {env}: old={p['n_old']} -> new={n_new} rows, "
            f"{p['old_eps']} -> {n_new_eps} episodes, "
            f"ep block={ep_start + k * 1000}..{ep_start + k * 1000 + 29}, "
            f"pair {pair_cursor + 1}..{pair_cursor + n_new}")
        # episode remap: 独立 1000 块,本地 0..29 原样平移（保持 %10 结构）
        ep_new = (ep_start + k * 1000) + fx["episode_id"].to(
            base["episode_id"].dtype)
        # pair remap: 当前全局最大 + 1 续接（对局部 id 做 dense 映射，鲁棒）
        loc = fx["pair_id"].tolist()
        m = {u: pair_cursor + 1 + i for i, u in enumerate(sorted(set(loc)))}
        pair_new = torch.tensor([m[v] for v in loc],
                                dtype=base["pair_id"].dtype)
        for key in base:
            if isinstance(base[key], torch.Tensor):
                if key == "episode_id":
                    v = ep_new
                elif key == "pair_id":
                    v = pair_new
                else:
                    v = fx[key]
                extra[key].append(v)
        extra_refs.extend(fx["frame_refs"])
        meta_src[src_idx] = fx["metadata"]["source_files"][0]
        pair_cursor += n_new
        del fx
    # assemble（键与顺序与 base 完全一致；dict 键在下方显式赋值）
    merged = {}
    for key in base:
        a = base[key]
        if isinstance(a, torch.Tensor):
            merged[key] = torch.cat([a[keep]] + extra[key], dim=0)
        elif isinstance(a, list):
            assert key == "frame_refs"
            merged[key] = [a[i] for i in range(n_base) if keep[i]] + extra_refs
        # dict（normalization / metadata）保持 base 键位置，下面显式设置
    meta = dict(base["metadata"])
    meta["source_files"] = meta_src
    meta["clean_resample"] = list(tasks)
    meta["n_trajectories"] = (
        base["metadata"]["n_trajectories"]
        - sum(p["old_eps"] for p in plan)
        + sum(p["new_eps"] for p in plan)
    )
    # 显式赋值保持 base 键序（normalization 在 metadata 之前）
    merged["normalization"] = base["normalization"]
    merged["metadata"] = meta
    return merged, plan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                    help="comma-separated env names (default: 11 final tasks)")
    ap.add_argument("--window-dir", default=DATA)
    ap.add_argument("--output", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="inspect base + window-file readiness only, never write")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in ENV_TO_TASK:
            raise ValueError(f"unknown task {t!r} (not in ENV_TO_TASK)")
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"duplicate tasks in --tasks: {tasks}")

    log(f"loading base: {args.base}")
    base = torch.load(args.base, map_location="cpu", weights_only=True)
    n_base = base["actions"].shape[0]
    log(f"base rows={n_base}  keys={list(base.keys())}")

    plan, _ = build_plan(base, tasks, args.window_dir)
    if args.dry_run:
        rc = print_dry_run(base, plan, n_base, args.base)
        del base
        return rc

    missing = [p for p in plan if not p["wready"]]
    if missing:
        log(f"ABORT: {len(missing)} window files not ready:")
        for p in missing:
            tag = "INVALID" if p["werr"] else "missing"
            log(f"  - {p['wpath']} [{tag}]" + (f" {p['werr']}" if p["werr"] else ""))
        log("wait for resample_highrisk_tasks.sh to finish (check "
            "logs/resample_summary.txt), then re-run. No file written.")
        return 1
    if os.path.exists(args.output):
        log(f"ABORT: refusing to overwrite existing output {args.output}")
        return 1
    # 轨迹文件也必须在（frame 解码校验要用）
    for p in plan:
        if not os.path.isfile(p["tpath"]):
            log(f"ABORT: missing trajectory file {p['tpath']}")
            return 1

    log(f"merging {len(tasks)} tasks ...")
    merged, plan = merge(base, plan, tasks)
    log(f"merged rows: {merged['actions'].shape[0]}")

    tmp = args.output + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    log(f"saving -> {tmp}")
    torch.save(merged, tmp)
    os.replace(tmp, args.output)
    log(f"saved {args.output} ({os.path.getsize(args.output)} bytes)")

    log("")
    log("=== post-merge validation ===")
    try:
        passed = validate_merged(merged, base, plan, tasks, args.output)
    except Exception as exc:
        log(f"VALIDATION CRASHED: {exc!r}; deleting output file")
        os.remove(args.output)
        raise
    if not passed:
        log("VALIDATION FAILED: deleting output file")
        os.remove(args.output)
        return 1
    log("")
    log(f"ALL CHECKS PASS: {args.output}")
    log(f"  total rows = {n_base} - {sum(p['n_old'] for p in plan)} + "
        f"{sum(p['n_new'] for p in plan)} = {merged['actions'].shape[0]}")
    log(f"  n_trajectories = {merged['metadata']['n_trajectories']}")
    log(f"  pair_id range (new) = {int(base['pair_id'].max()) + 1}..{int(merged['pair_id'].max())}")
    log(f"  episode blocks = {align1000(int(base['episode_id'].max()) + 1000)}.."
        f"{align1000(int(base['episode_id'].max()) + 1000) + len(tasks) * 1000 - 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
