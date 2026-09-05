from __future__ import annotations
import argparse
import math
from pathlib import Path
import torch
from va_compound.exact_resume import WMRM_DETACH_PROPOSAL_STAGE_STATE_MIGRATION, WMRM_WORLD_WEIGHT_1_TO_0_5_MIGRATION, WMRM_STATIC_CONSTRAINT_WEIGHT_4_TO_2_MIGRATION, WMRM_ACTION_RANK_CAP_NONE_TO_0_2_MIGRATION
from va_compound.world_contract import PEER_H15_P2_TO_P15_TEMPORAL_MIGRATION, PEER_H15_PREFIX_TAIL_FLOW_MIGRATION, PEER_H15_TO_H50_ACTION_MIGRATION, PEER_H50_ACTION_ONLY_TO_JOINT_MIGRATION, PEER_PLANNING_STRIDES, PEER_VA8_TO_VA16_CAPACITY_MIGRATION, PEER_WORLD8_TO_WORLD7_REPAIR_MIGRATION
from va_compound.world_supervision import canonical_stage_weight_overrides

RETIRED_ARGUMENT_DEFAULTS = {
    'action_vision_backbone': 'none',
    'action_vision_checkpoint': None,
    'action_vision_only': False,
    'c2_contract_every': 500,
    'c2_contract_rho6': 0.8,
    'c2_controller': False,
    'c2_lambda_c': 0.0,
    'c2_lambda_f': 0.1,
    'c2_lambda_r': 1.0,
    'c2_recovery_ratio': 0.25,
    'c2_unfreeze_stage_a': False,
    'c2_v6a': Path('data/mw_buttonpress_v6a.pt'),
    'c2_v6b': Path('data/mw_buttonpress_v6b.pt'),
    'capacity_new_only': False,
    'capacity_phase2_gates': False,
    'compile_every': 4,
    'compile_n_readout': 16,
    'compile_n_scene': 16,
    'compile_task': False,
    'dense_readout': False,
    'dense_readout_mtvj': False,
    'dino_dense_metric': False,
    'dino_roi_alpha': None,
    'dino_roi_checkpoint': None,
    'direct_head': False,
    'e2e_data': None,
    'e2e_pooling': 'flat',
    'evidence_tokens': 16,
    'evsm': False,
    'evsm_kappa': 0.02,
    'evsm_temp': 0.005,
    'flow_semantic': False,
    'fork_data': None,
    'fork_k': 83,
    'fork_skip_contract': False,
    'frame_aug': False,
    'frame_aug_geometric': True,
    'future_predict': False,
    'future_predict_weight': 0.0,
    'language_dtype': 'bfloat16',
    'language_max_length': 64,
    'live_root': Path('/media/ryan/robot-data/datasets/benchmark_data/raw/metaworld/lerobot_metaworld_mt50'),
    'live_vjepa': False,
    'local_slots_data': None,
    'local_slots_direct288': False,
    'local_slots_fixed_query': False,
    'lora_alpha': 32.0,
    'lora_lr': 0.0001,
    'lora_rank': 0,
    'lr_action_vision': 2e-05,
    'lr_mtvj_metric_head': 1e-06,
    'lr_mtvj_relation': 2e-05,
    'lr_servo': None,
    'lr_slot': None,
    'memory_split': False,
    'metric_geometry_inject': False,
    'metric_visual_checkpoint': None,
    'mtvj_roi_alpha': None,
    'mtvj_roi_checkpoint': None,
    'mtvj_train_metric_head': False,
    'mtvj_train_relation': False,
    'mtvj_visual_aux_batch': 8,
    'mtvj_visual_aux_every': 0,
    'mtvj_visual_aux_loc_lambda': 1.0,
    'mtvj_visual_aux_vis_lambda': 0.5,
    'multi_mode': False,
    'pair_loss_weight': 1.0,
    'pair_mode': 'shared_cf',
    'pair_probe_tau_max': 0.5,
    'pair_start_atol': 0.0,
    'pair_start_cosine': 0.0,
    'perturb_data': None,
    'phase_bins': 0,
    'phase_seed': 0,
    'plan_resampler': False,
    'qwen_lr': 1e-05,
    'qwen_unfreeze_blocks': 0,
    'replace_mtvj_metric_head_from_external': False,
    'role_query': False,
    'role_query_tokens': 16,
    'role_seeds': None,
    'sam_rho': 0.0,
    'scene_teacher': False,
    'semantic_act_grad_scale': 0.1,
    'semantic_adapter': False,
    'semantic_anchor_layers': '',
    'semantic_anchor_weight': 0.0,
    'semantic_geometry_weight': 0.0,
    'semantic_lora_rank': 8,
    'semantic_lora_suffixes': 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
    'semantic_top_layers': 4,
    'sequences_per_episode': 4,
    'servo': False,
    'servo_dls': False,
    'servo_lambda': 0.01,
    'servo_only': False,
    'servo_perturb_ratio': 0.5,
    'servo_rank': 2,
    'sliding_window': False,
    'success_only': False,
    'task35_precision_contract': False,
    'task_tokens': 8,
    'training_stage': None,
    'unfreeze_blocks': None,
    'vision_dtype': 'bfloat16',
    'vision_lr': 1e-05,
    'wmrm_adep_weight': 0.0,
    'wmrm_lang_align_weight': 0.0,
    'wmrm_stage_gate_start': None,
}

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
            description="MetaWorld DINO flow policy and peer World trainer"
        )
    parser.add_argument("--data", type=Path, help="optional paired precomputed .pt dataset")
    parser.add_argument(
            "--va-data",
            type=Path,
            help=(
                "peer_sync_h6 joint training: action/Flow dataset. It is sampled "
                "independently from --world-data on every optimizer step"
            ),
        )
    parser.add_argument(
            "--world-data",
            type=Path,
            help=(
                "peer_sync_h6 joint training: logged-transition World dataset. "
                "Its episodes must not overlap --va-data unless "
                "--peer-shared-full-data is explicit"
            ),
        )
    parser.add_argument(
            "--peer-shared-full-data",
            action="store_true",
            help=(
                "peer_sync_h6: let the independent VA and World loaders use one "
                "identical full-training payload instead of episode-disjoint payloads"
            ),
        )
    parser.add_argument(
            "--control-stride",
            type=int,
            default=6,
            help="决策点间隔（80 FPS 帧，--live-vjepa 用）：6=13.3Hz（v5 默认），"
            "2=40Hz，1=80Hz。须与数据提取时一致（与 payload 行数对齐）。",
        )
    parser.add_argument(
            "--planning-stride",
            type=int,
            default=6,
            help=(
                "每次动作块预测后实际执行的动作数 P；peer_sync_h6 支持 "
                "1/2/3/6/15，且必须等于 control-stride 和 flow-prefix-steps；"
                "World horizon 可等于 P 或完整动作 horizon"
            ),
        )
    parser.add_argument(
            "--deployment-execution-horizon",
            type=int,
            default=0,
            help=(
                "closed-loop actions executed before replanning; 0 uses planning-stride. "
                "H15/P15 checkpoints execute and train on the same 15-step cadence"
            ),
        )
    parser.add_argument(
            "--lr-vision",
            type=float,
            default=3e-6,
            help="解冻主视觉骨干的学习率（V-JEPA 或 DINO-main）",
        )
    parser.add_argument(
            "--vision-unfreeze-last",
            type=int,
            default=0,
            help="解冻 V-JEPA 最后 N 个 block（0 = 保持冻结；与 --vision-unfreeze-all 互斥）",
        )
    parser.add_argument(
            "--vision-unfreeze-all",
            action="store_true",
            help="全量解冻主视觉骨干（V-JEPA 或 DINO-main 的 stem/blocks/norm）",
        )
    parser.add_argument(
            "--qwen-keep-layers",
            type=int,
            default=0,
            help="physically keep the first N Qwen text layers; 0 keeps the full tower",
        )
    parser.add_argument(
            "--dual-attention",
            action="store_true",
            help="第二轮架构重构：非 sequential VA 层动作 query 拆 physical/semantic "
            "双注意力（sequential 层保持旧共享路径；与 --sequential-coupling=1 同开"
            "时报错——每层都是 sequential 双注意力永不生效；>1 时仅警告）",
        )
    parser.add_argument(
            "--wmrm",
            action="store_true",
            help="WAM4VA（原 --wmrm）：世界状态通过层间 K/V 与 VA 交换信息",
        )
    parser.add_argument(
            "--wam4va",
            action="store_true",
            dest="wmrm",
            help="WAM4VA：同 --wmrm。WAM 发布预测世界状态供下一层 VA 注意力读取",
        )
    parser.add_argument(
            "--wmrm-full-language-tokens",
            action="store_true",
            help=(
                "让 World 的每个 belief token 直接 masked-attend 同一份完整 Qwen "
                "token 序列；不再经固定 task queries 压成一个向量"
            ),
        )
    parser.add_argument(
            "--disable-runtime-integrity-checks",
            dest="runtime_integrity_checks",
            action="store_false",
            default=True,
            help=(
                "跳过 World 热路径中诊断性 finite/mask-content CUDA 检查；"
                "形状、dtype和训练计算不变（默认仍开启 fail-fast）"
            ),
        )
    parser.add_argument(
            "--va-world-mode",
            choices=("legacy", "peer_sync_h6"),
            default="legacy",
            help=(
                "VA/World action source: legacy requires the caller-provided executable "
                "action; peer_sync_h6 adds a deterministic H6 readout. Both use delayed "
                "bidirectional state exchange through VA attention K/V"
            ),
        )
    parser.add_argument(
            "--wmrm-target",
            choices=("dino", "vjepa", "metric"),
            default="dino",
            help="WAM 下一步监督：dino=下一决策完整 DINO latent map（与 VA 同周期）；"
            "vjepa=下一决策 H11 均值（冻塔白得空间）；metric=旧几何",
        )
    parser.add_argument(
            "--wmrm-world-weight",
            type=float,
            default=1.0,
            help="WMRM 世界预测 MSE 权重（仅 --wmrm；target 为下一决策 metric_g，stop-grad）",
        )
    parser.add_argument(
            "--wmrm-static-constraint-weight",
            type=float,
            default=4.0,
            help="visual World static-copy constraint loss weight (default: 4.0)",
        )
    parser.add_argument(
            "--wmrm-stage-s5-weight",
            type=float,
            default=None,
            help=(
                "replace stage-5 weight after decay/floor "
                "(default: 0.1; next-run S5/S6 experiment uses 0.5)"
            ),
        )
    parser.add_argument(
            "--wmrm-stage-s6-weight",
            type=float,
            default=None,
            help=(
                "replace stage-6 weight after decay/floor "
                "(default: 0.25; next-run S5/S6 experiment uses 1.0)"
            ),
        )
    parser.add_argument(
            "--wmrm-late-stage-anchor-weight",
            type=float,
            default=0.0,
            help=(
                "extra unnormalized S5/S6 World objective: "
                "weight * (0.5 * L_S5 + 1.0 * L_S6). "
                "Does not change the existing stage-mean weights. "
                "0 keeps the current loss graph"
            ),
        )
    parser.add_argument(
            "--wmrm-action-rank-per-sample-cap",
            type=float,
            default=None,
            help=(
                "cap each action-ranking sample before masked transition reduction "
                "(default: uncapped)"
            ),
        )
    parser.add_argument(
            "--visual-world-supervision",
            action="store_true",
            help="use visual-motion-aware full-map World supervision on logged actions",
        )
    parser.add_argument(
            "--world-split-manifest",
            type=Path,
            help="immutable episode-level train/eval split manifest for visual World runs",
        )
    parser.add_argument(
            "--world-action-rank-stage",
            choices=("final", "cycle"),
            default="cycle",
            help="v6 shuffled-action gap supervision at the final or rotating WAM stage",
        )
    parser.add_argument(
            "--wmrm-inject",
            choices=("last", "all", "even"),
            default="all",
            help="World state 更新层：all 每层；last 只末层；even 奇数层+末层",
        )
    parser.add_argument(
            "--wmrm-adep-margin",
            type=float,
            default=0.05,
            help="动作打乱后 world MSE 至少应增加的幅度",
        )
    parser.add_argument(
            "--wmrm-cycle-steps",
            type=int,
            default=6,
            dest="wmrm_cycle_steps",
            help="WAM 执行前缀步数（须与闭环 --execute-steps 一致）",
        )
    parser.add_argument(
            "--wmrm-detach-proposal-stage-state",
            action="store_true",
            help=(
                "在相邻 World stage 状态间 stop-grad；"
                "前向值不变，默认关闭以保留跨层梯度"
            ),
        )
    parser.add_argument(
            "--wmrm-map-size",
            type=int,
            default=16,
            dest="wmrm_map_size",
            help="无参平均池化把 DINO patch 压成 map_size×map_size 再预测（默认 16，对齐 DINO 网格）",
        )
    parser.add_argument(
            "--wmrm-map-channels",
            type=int,
            default=32,
            dest="wmrm_map_channels",
            help="DINO 空间图通道数",
        )
    parser.add_argument(
            "--wmrm-world-grid",
            type=int,
            default=16,
            dest="wmrm_world_grid",
            help="握手空间格子，默认 16=满 DINO 网格，不再 16→4 池化",
        )
    parser.add_argument(
            "--wmrm-predictor",
            choices=("legacy", "st_blocks"),
            default="legacy",
            dest="wmrm_predictor",
            help="世界图预测器：legacy=浅卷积；st_blocks=V-JEPA2 风格时空 Transformer",
        )
    parser.add_argument(
            "--wmrm-predictor-depth",
            type=int,
            default=6,
            dest="wmrm_predictor_depth",
        )
    parser.add_argument(
            "--wmrm-predictor-width",
            type=int,
            default=384,
            dest="wmrm_predictor_width",
        )
    parser.add_argument(
            "--wmrm-predictor-heads",
            type=int,
            default=12,
            dest="wmrm_predictor_heads",
        )
    parser.add_argument(
            "--wmrm-predictor-copies",
            type=int,
            default=1,
            dest="wmrm_predictor_copies",
        )
    parser.add_argument(
            "--wmrm-feature-metric",
            choices=("mse", "cosine"),
            default="mse",
            help="DINO-map error; cosine L2-normalizes channel directions",
        )
    parser.add_argument(
            "--wmrm-only",
            "--world-only",
            dest="wmrm_only",
            action="store_true",
            help="World 数据阶段：VA 只负责产生 detached 层消息，仅优化 WAM/World loss。",
        )
    parser.add_argument(
            "--va-only",
            action="store_true",
            help="动作数据阶段：WAM 只负责产生 detached 世界消息，仅优化 VA/Flow loss。",
        )
    parser.add_argument(
            "--feature-autocast-bf16",
            action="store_true",
            help="Run the feature-policy forward in CUDA BF16 autocast while keeping "
            "model/optimizer parameters and loss reductions in FP32.",
        )
    parser.add_argument(
            "--single-task",
            action="store_true",
            help="train Flow Matching without the unavailable multi-instruction pair loss",
        )
    parser.add_argument("--mode", choices=("bidir_va", "uni_a"), default="bidir_va")
    parser.add_argument(
            "--attention-variant",
            choices=("flat", "smc"),
            default="flat",
            help="shared-softmax attention variant: 'flat' (baseline) or 'smc' "
            "source-measure correction (log N_s subtracted per source before softmax)",
        )
    parser.add_argument(
            "--va-attention-backend",
            choices=("manual", "auto"),
            default="manual",
            help="VA shared attention implementation: manual materializes FP32 QK scores; "
            "auto uses fused SDPA on the compatible flat/non-dual path and falls back "
            "otherwise.",
        )
    parser.add_argument(
            "--action-query-cond",
            action="store_true",
            help="Qwen-conditioned action queries (2026-08-06 GPT 方案 A): language "
            "summary -> MLP -> per-horizon query offsets, zero-init so training starts "
            "identical to the static-query baseline",
        )
    parser.add_argument(
            "--sequential-coupling",
            type=int,
            default=0,
            help="every N-th VA layer uses sequential A->V/T->A coupling "
            "(0 = all-joint, legacy behavior; 2026-08-07 审阅落地④)",
        )
    parser.add_argument(
            "--flow-cond",        choices=("entry", "adaln"),
            default="entry",
            help="flow head conditioning: entry (legacy, add at input) or adaln "
            "(per-layer AdaLN-Zero + cross-attention; 2026-08-07)",
        )
    parser.add_argument(
            "--va-last3-cross-attn",
            action="store_true",
            help="Evo-1-style three-branch cross-attention over the final three VA "
            "action streams before Flow; zero-gated at initialization",
        )
    parser.add_argument(
            "--dino-qwen-cross-modal-bridge",
            action="store_true",
            help="zero-gated bidirectional DINO/Qwen cross-attention before VA",
        )
    parser.add_argument(
            "--flow-layers",
            type=int,
            default=2,
            help="flow head transformer layers (π0-style expert 加厚用；"
            "resume 时新层随机初始化，已有层继承)",
        )
    parser.add_argument(
            "--action-vision-encode-batch",
            type=int,
            default=4,
            help="Frozen action-tower image microbatch (4 is the safe ViT-L default "
            "for the 16-GiB training GPU).",
        )
    parser.add_argument(
            "--dino-main-vision",
            action="store_true",
            help="DINO-main replacement：DINOv2 替换 V-JEPA 作为 VA 主视觉骨干；"
            "默认冻结，--vision-unfreeze-all 可全量训练。",
        )
    parser.add_argument(
            "--main-vision-checkpoint",
            type=Path,
            default=None,
            help="DINO-main 视觉塔的本地 timm 权重（--dino-main-vision 必填）。",
        )
    parser.add_argument(
            "--main-vision-encode-batch",
            type=int,
            default=16,
            help="DINO-main 图像编码 microbatch。",
        )
    parser.add_argument(
            "--main-vision-grid",
            type=int,
            default=8,
            help="DINO-main 每帧 16x16 patch 网格池化到 grid x grid（默认 8 → 64 帧内 token）。",
        )
    parser.add_argument(
            "--main-vision-frames",
            type=int,
            default=4,
            help="DINO-main 每决策消费的窗口帧数（默认 4 = [d-6,d-4,d-2,d]）。",
        )
    parser.add_argument(
            "--main-vision-temporal",
            action="store_true",
            help="为 frame-major DINO patch tokens 加 learned 四帧 slot embedding；"
            "打破旧路径对 [d-6,d-4,d-2,d] 顺序的集合置换不变性。",
        )
    parser.add_argument(
            "--main-vision-temporal-scale",
            type=float,
            default=1.0,
            help="learned frame embedding 的乘法 gate（训练默认 1；0 仅用于因果消融）。",
        )
    parser.add_argument(
            "--longtraj-dir",
            type=Path,
            default=None,
            help="longtraj JPEG 帧文件所在目录（默认仓库 data/）。扩产数据把分片合进"
            "每任务一个文件后放在别的目录时用它指过去；exact-resume 的帧指纹随之"
            "跟到同一目录，不会去 data/ 下认错旧文件。",
        )
    parser.add_argument(
            "--online-episode-sampling",
            action="store_true",
            help=(
                "treat peer --va-data/--world-data (or --va-only --data) as a "
                "full-episode JSON index and "
                "generate arbitrary overlapping H15/H50 samples at batch time; no "
                "offline action/frame windows are loaded"
            ),
        )
    parser.add_argument(
            "--online-episode-samples",
            type=int,
            default=6,
            help="independent random starts drawn per full episode per epoch (default 6)",
        )
    parser.add_argument(
            "--online-action-horizon",
            type=int,
            choices=(15, 50),
            default=15,
            help="runtime action labels sampled from full episodes; World remains +15",
        )
    parser.add_argument(
            "--online-recovery-samples-per-episode",
            type=int,
            default=0,
            help=(
                "for perturbed episodes, reserve this many online starts for visible "
                "expert recovery; remaining starts prefer clean trajectory regions"
            ),
        )
    parser.add_argument(
            "--longtraj-decode-cache-tasks",
            type=int,
            default=None,
            help="常驻内存的已解码任务数（默认沿用按模式推断的值）。整任务文件全量"
            "解码，任务切换即重解码；数据扩产后单任务可达数十 GB 且解码要几分钟，"
            "设为任务总数可用内存换掉这笔反复开销。",
        )
    parser.add_argument(
            "--dino-feature-cache",
            type=Path,
            default=None,
            help="DINO-main/DINO-metric 预计算特征缓存目录（scripts/"
            "build_dino_feature_cache.py 生成；block11/block23 fp16 memmap；"
            "task35 ROI 精插还要求 exact raw_frames.npy）。"
            "冻结塔在线编码占步时 84%%，缓存读把 13000 步从 ~9.4h 降到 ~2.5h；"
            "位级一致性由预计算脚本内置 torch.equal 验证，eval 仍在线编码。",
        )
    parser.add_argument(
            "--slot-free-policy",
            action="store_true",
            help=(
                "正式无槽策略契约：禁止 metric_g、metric relation tokens 和固定 "
                "local-slot reader 进入 VA/World/action"
            ),
        )
    parser.add_argument(
            "--lang-fixed-vector",
            action="store_true",
            help="grounding 对照（Codex 2026-08-08）：语言通道替换为数据集全局均值常量向量，"
            "重训同容量模型——完整模型 vs 固定语言基线的差距即语言条件的因果贡献。"
            "仅 feature 路径（非 live）可用。",
        )
    parser.add_argument(
            "--qk-norm",
            action="store_true",
            help="per-head RMSNorm on Q/K in the VA coupling layers (Su Shen QK-Norm)",
        )
    parser.add_argument(
            "--vision-pooling",
            choices=("flat", "spatial", "spatiotemporal"),
            default="flat",
            help="vision feature variant: 'flat' (A), 'spatial' (B), or "
            "'spatiotemporal' (ST288/live 288-token；仅影响 training_contract 记录，"
            "供闭环评估对齐在线池化)",
        )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="must be even")
    parser.add_argument(
            "--num-workers",
            type=int,
            default=0,
            help="DataLoader worker 数（--live-vjepa 帧解码是 CPU 瓶颈：单线程冷解码 "
            "64 帧 ≈ 1.7s/step 而 GPU 仅忙 ~0.5s；num_workers>=4 与 GPU 重叠解码，"
            "step 时间降至 GPU 上限。0 = 现状主进程串行）",
        )
    parser.add_argument(
            "--peer-batch-prefetch",
            action="store_true",
            help=(
                "peer 双数据流用一个后台线程有界预取 VA+World batch；默认关闭。"
                "epoch 边界会等成功更新推进 sampler 后再重启 iterator"
            ),
        )
    parser.add_argument(
            "--peer-batch-prefetch-depth",
            type=int,
            default=1,
            help="peer 后台预取的最大联合 batch 数（默认 1）",
        )
    parser.add_argument(
            "--task-sampling",
            choices=("uniform", "balanced", "weighted", "full", "mixed"),
            default="uniform",
            help="难度分层采样（E7 用 weighted，2026-08-09）：按 instruction_id → "
            "MT50 难度权重（easy 0.5/med 1.0/hard 2.0/vh 3.0，scripts/mt50_difficulty.py）"
            "多项式抽样，困难任务过采样、简单任务降采样；balanced = 每个 epoch "
            "严格均衡所有活跃任务；full = 每 epoch 无放回完整遍历所有行；"
            "mixed = 每个 batch 均匀混合多个任务并加入固定 replay anchor；"
            "uniform = 按数据行均匀采样（会继承窗口数偏置）",
        )
    parser.add_argument(
            "--mixed-tasks-per-batch",
            type=int,
            default=4,
            help="--task-sampling mixed 时每个 global batch 的不同任务数",
        )
    parser.add_argument(
            "--anchor-replay-fraction",
            type=float,
            default=0.25,
            help="mixed batch 内固定 epoch-0 replay anchor 比例",
        )
    parser.add_argument(
            "--pcgrad",
            action="store_true",
            help="对 mixed batch 内各任务的动作梯度执行冲突投影后再加入 World 梯度",
        )
    parser.add_argument(
            "--pcgrad-separate-world",
            action="store_true",
            help=(
                "Action 与 World 分别按任务执行 PCGrad；仅在共享 DINO 梯度冲突时"
                "投影 World 梯度，VA/World 私有表征不做跨分支投影"
            ),
        )
    parser.add_argument(
            "--zero-redundancy-optimizer",
            action="store_true",
            help="在多 GPU 间分片 AdamW 状态；参数和梯度语义不变",
        )
    parser.add_argument(
            "--task-locality-block-batches",
            type=int,
            default=16,
            help="MT-VJ weighted/balanced sampler 每个同任务块的 batch 数（默认 16；"
            "解码切换成为瓶颈时可调到 32）；full 固定保持整个任务连续。",
        )
    parser.add_argument("--sequence-length", type=int, default=4, help="synthetic BPTT length")
    parser.add_argument("--min-sequence-length", type=int, default=4)
    parser.add_argument("--min-pair-action-delta", type=float, default=1e-3)
    parser.add_argument("--flow-steps", type=int, default=8, help="deployment Euler steps")
    parser.add_argument(
            "--flow-prefix-steps",
            type=int,
            default=6,
            help="flow loss/diagnostic split; deployment cadence is configured separately.",
        )
    parser.add_argument(
            "--flow-prefix-weight",
            type=float,
            default=1.0,
            help="flow 前缀逐元素 MSE 权重（默认 1.0，保持旧行为）。",
        )
    parser.add_argument(
            "--flow-tail-weight",
            type=float,
            default=1.0,
            help="flow 尾部逐元素 MSE 权重（默认 1.0；H48 的6步前缀若希望约80/20 "
            "总权重，尾部应约0.036，而非0.1）。",
        )
    parser.add_argument(
            "--va-layers",
            type=int,
            default=4,
            help="VACouplingLayer count in the decision stack (depth probe)",
        )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
            "--lr-wmrm-predictor",
            type=float,
            default=None,
            help=(
                "AdamW learning rate for wmrm.st_predictor only "
                "(default: share --lr). Use 3e-5 to slow the shared 6-block residual"
            ),
        )
    parser.add_argument(
            "--wmrm-predictor-grad-clip",
            type=float,
            default=None,
            help=(
                "clip wmrm.st_predictor independently of the rest of the main model "
                "(default: share the global 1.0 clip)"
            ),
        )
    parser.add_argument(
            "--lr-va",
            type=float,
            default=None,
            help="PULSE-VA：共享 VA/头 LR（默认 = --lr）；Codex Stage A 建议 3e-5",
        )
    parser.add_argument(
            "--head-only",
            action="store_true",
            help="Stage 1 对齐模式：只训练 flow head（VA/槽/V-JEPA 全部冻结）"
            "——随机初始化的动作头噪声梯度不污染已训练的视觉/集成参数；"
            "Stage 2 再去掉本开关全量微调。",
        )
    parser.add_argument(
            "--prev-dropout",
            type=float,
            default=0.0,
            help="probability of zeroing previous_action per training sample (0 = off). "
            "P0-1 closed-loop prev self-excitation contract fix (2026-08-06 Codex): "
            "training uses teacher-forced prev, deployment uses the model's own output; "
            "dropout aligns the first-decision prev=0 condition. Features path only.",
        )
    parser.add_argument(
            "--max-gradient-norm",
            type=float,
            default=None,
            help="abort an update when the aggregate gradient norm exceeds this threshold "
            "(default: disabled). Individual finite gradient elements may exceed it. "
            "This argument is bound into the exact-resume contract.",
        )
    parser.add_argument("--seed", type=int, default=0)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
            "--resume",
            type=Path,
            help="load model weights from a checkpoint; optimizer/sampler/RNG restart",
        )
    resume_group.add_argument(
            "--resume-exact",
            type=Path,
            help="strictly continue model, AdamW, TaskLocality sampler, step, and RNG state",
        )
    resume_group.add_argument(
            "--resume-weights",
            type=Path,
            help=(
                "load model/MT-VJ weights only; optimizer, sampler, RNG and step restart. "
                "Allowed with --task35-precision-contract so a new data/cache SHA can be stamped"
            ),
        )
    parser.add_argument(
            "--resume-exact-contract-migration",
            choices=[
                WMRM_DETACH_PROPOSAL_STAGE_STATE_MIGRATION,
                WMRM_WORLD_WEIGHT_1_TO_0_5_MIGRATION,
                WMRM_STATIC_CONSTRAINT_WEIGHT_4_TO_2_MIGRATION,
                WMRM_ACTION_RANK_CAP_NONE_TO_0_2_MIGRATION,
            ],
            help=(
                "allow one named, controlled exact-contract compatibility transition; "
                "the selector is operational and is not saved as a run semantic"
            ),
        )
    parser.add_argument(
            "--resume-weights-migration",
            choices=[
                PEER_WORLD8_TO_WORLD7_REPAIR_MIGRATION,
                PEER_H15_PREFIX_TAIL_FLOW_MIGRATION,
                PEER_H15_P2_TO_P15_TEMPORAL_MIGRATION,
                PEER_H15_TO_H50_ACTION_MIGRATION,
                PEER_H50_ACTION_ONLY_TO_JOINT_MIGRATION,
                PEER_VA8_TO_VA16_CAPACITY_MIGRATION,
            ],
            help=(
                "one named peer weights migration: legacy H6-to-H15 World repair, "
                "s1752 H15-to-isolated-prefix/tail Flow, H15 P2-to-P15 cadence, "
                "H15-to-H50 nested action expansion, H50 action-only-to-joint, "
                "or gated 8-to-16 VA capacity expansion"
            ),
        )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", type=Path)
    parser.add_argument(
            "--save-every",
            type=int,
            default=0,
            help="periodically overwrite --save every N steps (atomic tmp+rename); "
            "0 disables periodic saves (crash loses the whole run)",
        )
    parser.add_argument(
            "--save-step-copies",
            action="store_true",
            help="also write --save stem_s{step}.pt on each periodic save",
        )
    parser.add_argument("--wmrm-adep-weight", type=float, choices=(0.0,), default=0.0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for key, value in RETIRED_ARGUMENT_DEFAULTS.items():
        setattr(args, key, value)
    return args


def visual_world_stage_weight_overrides(args: argparse.Namespace) -> dict[int, float]:
    """S5/S6 overrides for the next-run weight experiment; empty keeps the old schedule."""
    overrides: dict[int, float] = {}
    s5 = getattr(args, "wmrm_stage_s5_weight", None)
    s6 = getattr(args, "wmrm_stage_s6_weight", None)
    if s5 is not None:
        overrides[5] = float(s5)
    if s6 is not None:
        overrides[6] = float(s6)
    return canonical_stage_weight_overrides(overrides)

def validate_args(args: argparse.Namespace) -> None:
    """Validate training arguments without loading data or models."""
    finite_positive = {"--lr": args.lr}
    for flag, value in finite_positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{flag} must be a positive finite value")
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if not (args.single_task and args.dino_main_vision and args.slot_free_policy
            and args.va_world_mode == "peer_sync_h6"):
        raise ValueError("MetaWorld trainer requires slot-free DINO peer training with --single-task")
    if args.resume_weights_migration not in {None, PEER_H15_TO_H50_ACTION_MIGRATION, PEER_H50_ACTION_ONLY_TO_JOINT_MIGRATION}:
        raise ValueError("this historical training migration is retired")
    if args.steps <= 0 or args.flow_steps <= 0:
        raise ValueError("training steps and flow steps must be positive")
    if int(getattr(args, "online_episode_samples", 6)) < 1:
        raise ValueError("--online-episode-samples must be positive")
    recovery_samples = int(
        getattr(args, "online_recovery_samples_per_episode", 0)
    )
    if not 0 <= recovery_samples <= int(args.online_episode_samples):
        raise ValueError(
            "--online-recovery-samples-per-episode must be in "
            "[0, --online-episode-samples]"
        )
    if args.deployment_execution_horizon < 0:
        raise ValueError("--deployment-execution-horizon must be non-negative")
    if not math.isfinite(args.pair_loss_weight) or args.pair_loss_weight < 0.0:
        raise ValueError("pair loss weight must be a non-negative finite value")
    if args.max_gradient_norm is not None and (
        not math.isfinite(args.max_gradient_norm) or args.max_gradient_norm <= 0.0
    ):
        raise ValueError("--max-gradient-norm must be a positive finite value")
    cap = args.wmrm_action_rank_per_sample_cap
    if cap is not None and (
        not math.isfinite(cap) or cap <= 0.0
    ):
        raise ValueError(
            "--wmrm-action-rank-per-sample-cap must be a positive finite value"
        )
    if cap is not None and not getattr(args, "visual_world_supervision", False):
        raise ValueError(
            "--wmrm-action-rank-per-sample-cap only applies with "
            "--visual-world-supervision"
        )
    if args.resume_exact_contract_migration is not None and args.resume_exact is None:
        raise ValueError(
            "--resume-exact-contract-migration requires --resume-exact"
        )
    if args.resume_weights_migration is not None and args.resume_weights is None:
        raise ValueError(
            "--resume-weights-migration requires --resume-weights"
        )
    peer_world = getattr(args, "va_world_mode", "legacy") == "peer_sync_h6"
    peer_va_only = peer_world and bool(getattr(args, "va_only", False))
    stage_gate_start = getattr(args, "wmrm_stage_gate_start", None)
    if stage_gate_start is not None and (
        not peer_world or not 0 < int(stage_gate_start) < int(args.va_layers) - 1
    ):
        raise ValueError(
            "--wmrm-stage-gate-start must be inside the peer World stage range"
        )
    if args.resume_weights_migration is not None and not peer_world:
        raise ValueError(
            "--resume-weights-migration requires --va-world-mode peer_sync_h6"
        )
    if (
        args.resume_weights_migration == PEER_VA8_TO_VA16_CAPACITY_MIGRATION
        and (
            int(args.va_layers) != 16
            or int(getattr(args, "wmrm_stage_gate_start", -1) or -1) != 7
            or int(getattr(args, "wmrm_predictor_depth", 0)) != 7
            or int(getattr(args, "wmrm_predictor_copies", 0)) != 11
            or getattr(args, "wmrm_feature_metric", "mse") != "cosine"
        )
    ):
        raise ValueError(
            f"{PEER_VA8_TO_VA16_CAPACITY_MIGRATION} requires "
            "--va-layers 16 --wmrm-stage-gate-start 7 "
            "--wmrm-predictor-depth 7 --wmrm-predictor-copies 11 "
            "--wmrm-feature-metric cosine"
        )
    va_data = getattr(args, "va_data", None)
    world_data = getattr(args, "world_data", None)
    shared_full_data = bool(getattr(args, "peer_shared_full_data", False))
    if (va_data is None) != (world_data is None):
        raise ValueError("--va-data and --world-data must be provided together")
    dual_peer_data = peer_world and va_data is not None and world_data is not None
    primary_data = va_data if dual_peer_data else args.data
    if getattr(args, "online_episode_sampling", False):
        if not dual_peer_data and not (peer_va_only and args.data is not None):
            raise ValueError(
                "--online-episode-sampling requires peer --va-data/--world-data "
                "or peer --va-only with --data"
            )
        if int(getattr(args, "planning_stride", 6)) != 15:
            raise ValueError("online full-episode sampling currently requires P15")
        if getattr(args, "dino_feature_cache", None) is not None:
            raise ValueError(
                "online random starts cannot use a fixed --dino-feature-cache"
            )
    elif int(getattr(args, "online_action_horizon", 15)) != 15:
        raise ValueError("--online-action-horizon requires --online-episode-sampling")
    if int(getattr(args, "online_action_horizon", 15)) == 50:
        stage1_entry = peer_va_only and (
            args.resume_weights_migration == PEER_H15_TO_H50_ACTION_MIGRATION
            or args.resume_exact is not None
        )
        joint_entry = dual_peer_data and not peer_va_only and (
            args.resume_weights_migration
            == PEER_H50_ACTION_ONLY_TO_JOINT_MIGRATION
            or args.resume_exact is not None
        )
        required_h50 = {
            "H50 Stage1 or joint resume contract": stage1_entry or joint_entry,
            "--planning-stride 15": int(args.planning_stride) == 15,
            "--deployment-execution-horizon 15": int(
                args.deployment_execution_horizon
            ) == 15,
            "--wmrm-cycle-steps 15": int(args.wmrm_cycle_steps) == 15,
            "--flow-prefix-steps 15": int(args.flow_prefix_steps) == 15,
        }
        missing_h50 = [name for name, enabled in required_h50.items() if not enabled]
        if missing_h50:
            raise ValueError(
                "H50/P15 training missing required settings: "
                + ", ".join(missing_h50)
            )
    if int(getattr(args, "peer_batch_prefetch_depth", 1)) < 1:
        raise ValueError("--peer-batch-prefetch-depth must be positive")
    if getattr(args, "peer_batch_prefetch", False) and not dual_peer_data:
        raise ValueError(
            "--peer-batch-prefetch requires peer_sync_h6 --va-data/--world-data"
        )
    if not peer_world and (va_data is not None or world_data is not None):
        raise ValueError("--va-data/--world-data are only supported by peer_sync_h6")
    if shared_full_data and not peer_world:
        raise ValueError("--peer-shared-full-data requires --va-world-mode peer_sync_h6")
    if peer_world:
        planning_stride = int(getattr(args, "planning_stride", 6))
        deployment_horizon = int(
            getattr(args, "deployment_execution_horizon", 0)
            or planning_stride
        )
        if not dual_peer_data and not (peer_va_only and args.data is not None):
            raise ValueError(
                "peer_sync_h6 requires joint --va-data/--world-data or "
                "Stage1 --va-only with --data"
            )
        if args.data is not None and not peer_va_only:
            raise ValueError(
                "peer_sync_h6 dual-stream training uses --va-data/--world-data, "
                "not --data"
            )
        if peer_va_only and dual_peer_data:
            raise ValueError("peer --va-only uses one --data stream, not VA/World streams")
        if (
            dual_peer_data
            and not shared_full_data
            and va_data.expanduser().resolve(strict=False)
            == world_data.expanduser().resolve(strict=False)
        ):
            raise ValueError("--va-data and --world-data must be different files")
        if getattr(args, "wmrm_only", False):
            raise ValueError(
                "peer_sync_h6 does not support --world-only"
            )
        required = {
            "--wam4va": bool(getattr(args, "wmrm", False)),
            "--visual-world-supervision": bool(
                getattr(args, "visual_world_supervision", False)
            ) or peer_va_only,
            "--planning-stride 1/2/3/6/15": planning_stride
            in PEER_PLANNING_STRIDES,
            "--control-stride == --planning-stride": int(
                getattr(args, "control_stride", 6)
            )
            == planning_stride,
            "--wmrm-cycle-steps == execution prefix or H15": int(
                getattr(args, "wmrm_cycle_steps", 0)
            )
            in {planning_stride, 15},
            "--deployment-execution-horizon == P or World horizon": (
                deployment_horizon
                in {
                    planning_stride,
                    int(getattr(args, "wmrm_cycle_steps", 0)),
                }
            ),
            "--flow-prefix-steps == --planning-stride": int(
                getattr(args, "flow_prefix_steps", 0)
            )
            == planning_stride,
            "--wmrm-inject all": getattr(args, "wmrm_inject", None) == "all",
            "differentiable World stage recurrence": not bool(
                getattr(args, "wmrm_detach_proposal_stage_state", False)
            ),
            "--sam-rho 0": float(getattr(args, "sam_rho", 0.0)) == 0.0,
            "no --e2e-data": args.e2e_data is None,
            "no --live-vjepa": not bool(args.live_vjepa),
            "no --dino-feature-cache": getattr(args, "dino_feature_cache", None)
            is None,
            "no --dense-readout-mtvj": not bool(args.dense_readout_mtvj),
            "no --action-vision-backbone": getattr(
                args, "action_vision_backbone", "none"
            )
            == "none",
            "no --perturb-data/--fork-data": args.perturb_data is None
            and args.fork_data is None,
            "--task-sampling balanced/weighted/full/mixed": args.task_sampling
            in {"balanced", "weighted", "full", "mixed"},
        }
        missing = [name for name, enabled in required.items() if not enabled]
        if missing:
            raise ValueError(
                "--va-world-mode peer_sync_h6 missing required settings: "
                + ", ".join(missing)
            )
        if args.resume is not None:
            raise ValueError(
                "peer_sync_h6 rejects legacy --resume; use scratch, "
                "--resume-weights, or --resume-exact"
            )
        if getattr(args, "resume_exact", None) is not None:
            peer_checkpoint = torch.load(
                args.resume_exact, map_location="cpu", weights_only=True
            )
            saved_mode = (peer_checkpoint.get("config") or {}).get(
                "va_world_mode", "legacy"
            )
            if saved_mode != "peer_sync_h6":
                raise ValueError(
                    "--resume-exact peer_sync_h6 requires a peer_sync_h6 checkpoint; "
                    f"checkpoint mode is {saved_mode!r}"
                )
    visual_world = bool(getattr(args, "visual_world_supervision", False))
    split_manifest = getattr(args, "world_split_manifest", None)
    if visual_world:
        required = {
            "--world-data" if dual_peer_data else "--data": (
                world_data is not None if dual_peer_data else args.data is not None
            ),
            "--world-split-manifest": split_manifest is not None,
            "--wam4va": bool(getattr(args, "wmrm", False)),
            "--wmrm-target dino": getattr(args, "wmrm_target", None) == "dino",
            "--wmrm-cycle-steps matches peer horizon": (
                int(getattr(args, "wmrm_cycle_steps", 0))
                in (
                    {int(getattr(args, "planning_stride", 6)), 15}
                    if peer_world else {6}
                )
            ),
            "--wmrm-inject all": getattr(args, "wmrm_inject", None) == "all",
            "VA/World depth contract": (
                int(getattr(args, "va_layers", 0)) == 8
                and stage_gate_start is None
            ) or (
                int(getattr(args, "va_layers", 0)) == 16
                and int(stage_gate_start or -1) == 7
            ),
            "--wmrm-predictor st_blocks": getattr(args, "wmrm_predictor", None)
            == "st_blocks",
            "WM predictor capacity contract": (
                int(getattr(args, "wmrm_predictor_depth", 0)) == 6
                and int(getattr(args, "wmrm_predictor_copies", 0)) == 1
                and int(getattr(args, "va_layers", 0)) == 8
            ) or (
                int(getattr(args, "wmrm_predictor_depth", 0)) == 7
                and int(getattr(args, "wmrm_predictor_copies", 0)) == 11
                and int(getattr(args, "va_layers", 0)) == 16
            ),
            "--wmrm-predictor-width 384": int(
                getattr(args, "wmrm_predictor_width", 0)
            )
            == 384,
            "--wmrm-predictor-heads 12": int(
                getattr(args, "wmrm_predictor_heads", 0)
            )
            == 12,
            "--wmrm-map-size 16": int(getattr(args, "wmrm_map_size", 0)) == 16,
            "--wmrm-map-channels 1024": int(
                getattr(args, "wmrm_map_channels", 0)
            )
            == 1024,
            "--wmrm-world-grid 16": int(getattr(args, "wmrm_world_grid", 0))
            == 16,
            "--dino-main-vision": bool(getattr(args, "dino_main_vision", False)),
            "--main-vision-grid 16": int(getattr(args, "main_vision_grid", 0)) == 16,
            "--main-vision-frames 4": int(getattr(args, "main_vision_frames", 0))
            == 4,
            "--sequence-length 4": int(getattr(args, "sequence_length", 0)) == 4,
            "--min-sequence-length 4": int(
                getattr(args, "min_sequence_length", 0)
            )
            == 4,
            "--single-task joint sampler": bool(args.single_task),
        }
        missing = [name for name, enabled in required.items() if not enabled]
        if missing:
            raise ValueError(
                "--visual-world-supervision missing required settings: "
                + ", ".join(missing)
            )
        if args.resume is not None:
            raise ValueError(
                "visual World training rejects legacy --resume; use scratch, "
                "--resume-weights, or --resume-exact"
            )
    elif split_manifest is not None:
        raise ValueError(
            "--world-split-manifest requires --visual-world-supervision"
        )
    flow_prefix_steps = getattr(args, "flow_prefix_steps", 6)
    flow_prefix_weight = getattr(args, "flow_prefix_weight", 1.0)
    flow_tail_weight = getattr(args, "flow_tail_weight", 1.0)
    if flow_prefix_steps <= 0:
        raise ValueError("--flow-prefix-steps must be positive")
    if flow_prefix_weight < 0.0 or flow_tail_weight < 0.0:
        raise ValueError("--flow-prefix-weight/--flow-tail-weight must be non-negative")
    if flow_prefix_weight == 0.0 and flow_tail_weight == 0.0:
        raise ValueError("flow prefix and tail weights cannot both be zero")
    replace_metric_head = getattr(
        args, "replace_mtvj_metric_head_from_external", False
    )
    action_vision = getattr(args, "action_vision_backbone", "none")
    action_vision_only = getattr(args, "action_vision_only", False)
    action_vision_checkpoint = getattr(args, "action_vision_checkpoint", None)
    if action_vision_only and (
        getattr(args, "head_only", False)
        or getattr(args, "servo_only", False)
        or getattr(args, "c2_controller", False)
        or getattr(args, "mtvj_train_relation", False)
        or getattr(args, "mtvj_train_metric_head", False)
    ):
        raise ValueError(
            "--action-vision-only freezes the base policy/metric path and is "
            "incompatible with head/servo/C2 or MT-VJ joint-training flags"
        )
    # DINO-main replacement（2026-08-14 用户决策）：V-JEPA/dense/metric 全部
    # 保留在代码中（flag 关闭即禁用，不删除），此处只校验组合合法性。
    dino_main_vision = bool(getattr(args, "dino_main_vision", False))
    if dino_main_vision:
        if action_vision_only:
            raise ValueError("--dino-main-vision is incompatible with --action-vision-only")
        if args.main_vision_checkpoint is None:
            raise ValueError("--dino-main-vision requires --main-vision-checkpoint")
        if not args.main_vision_checkpoint.expanduser().is_file():
            raise FileNotFoundError(
                f"main vision checkpoint does not exist: {args.main_vision_checkpoint}"
            )
        if args.main_vision_encode_batch < 1:
            raise ValueError("--main-vision-encode-batch must be positive")
        if args.vision_unfreeze_last:
            raise ValueError(
                "DINO-main currently supports full unfreezing only; use "
                "--vision-unfreeze-all"
            )
        if args.vision_unfreeze_all and (
            not math.isfinite(float(args.lr_vision)) or args.lr_vision <= 0.0
        ):
            raise ValueError("trainable DINO-main requires positive --lr-vision")
        if args.dino_feature_cache is not None:
            if args.vision_unfreeze_all:
                raise ValueError(
                    "trainable DINO-main cannot use a frozen --dino-feature-cache"
                )
            cache_dir = args.dino_feature_cache.expanduser()
            if not cache_dir.is_dir():
                raise ValueError(
                    f"--dino-feature-cache directory missing: {cache_dir}"
                )
            for name in ("meta.json", "index.pkl", "block23.npy", "block11.npy"):
                if not (cache_dir / name).exists():
                    raise ValueError(
                        f"--dino-feature-cache 缺少 {name}: {cache_dir}"
                    )
    elif getattr(args, "main_vision_checkpoint", None) is not None:
        raise ValueError("--main-vision-checkpoint requires --dino-main-vision")
    if getattr(args, "dino_feature_cache", None) is not None and not dino_main_vision:
        raise ValueError("--dino-feature-cache requires --dino-main-vision")
    if getattr(args, "main_vision_temporal", False) and not dino_main_vision:
        raise ValueError("--main-vision-temporal requires --dino-main-vision")
    if not math.isfinite(float(getattr(args, "main_vision_temporal_scale", 1.0))):
        raise ValueError("--main-vision-temporal-scale must be finite")
    if args.single_task and args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if getattr(args, "task_locality_block_batches", 16) <= 0:
        raise ValueError("--task-locality-block-batches must be positive")
    if args.task_sampling == "mixed":
        if not getattr(args, "online_episode_sampling", False):
            raise ValueError("--task-sampling mixed requires --online-episode-sampling")
        if args.mixed_tasks_per_batch < 2:
            raise ValueError("--mixed-tasks-per-batch must be at least 2")
        if not 0.0 < args.anchor_replay_fraction < 1.0:
            raise ValueError("--anchor-replay-fraction must be in (0,1)")
    if args.pcgrad:
        forbidden = {
            "--task-sampling mixed": args.task_sampling != "mixed",
            "peer data stream": not (
                (args.va_data is not None and args.world_data is not None)
                or (peer_va_only and args.data is not None)
            ),
            "no --direct-head": bool(args.direct_head),
            "no --sam-rho": float(args.sam_rho) != 0.0,
            "no semantic adapter": bool(args.semantic_adapter),
            "no future predictor": bool(args.future_predict)
            or float(args.future_predict_weight) != 0.0,
            "no visual auxiliary": int(args.mtvj_visual_aux_every) != 0,
            "no fork/perturb/C2": args.fork_data is not None
            or args.perturb_data is not None
            or bool(args.c2_controller),
        }
        active = [name for name, invalid in forbidden.items() if invalid]
        if active:
            raise ValueError("--pcgrad incompatible settings: " + ", ".join(active))
    if getattr(args, "pcgrad_separate_world", False):
        if not args.pcgrad:
            raise ValueError("--pcgrad-separate-world requires --pcgrad")
        if not dual_peer_data or not getattr(args, "visual_world_supervision", False):
            raise ValueError(
                "--pcgrad-separate-world requires joint VA/World streams with "
                "--visual-world-supervision"
            )
    if getattr(args, "resume_exact", None) is not None:
        if args.num_workers != 0:
            raise ValueError("--resume-exact requires --num-workers 0 (worker RNG is not checkpointed)")
        if not (
            primary_data is not None
            and args.single_task
            and args.task_sampling in {"weighted", "balanced", "full", "mixed"}
            and (
                args.dense_readout_mtvj
                or getattr(args, "dino_main_vision", False)
            )
        ):
            raise ValueError(
                "--resume-exact currently requires the single-task "
                "weighted/balanced/full/mixed "
                "MT-VJ or DINO-main data path (TaskLocalityWeightedSampler or "
                "TaskWeightedSampler)"
            )
    if getattr(args, "wmrm_full_language_tokens", False) and not getattr(
        args, "wmrm", False
    ):
        raise ValueError("--wmrm-full-language-tokens requires --wam4va")
    forbidden_slots = {
        "--dino-dense-metric": bool(getattr(args, "dino_dense_metric", False)),
        "--metric-geometry-inject": bool(
            getattr(args, "metric_geometry_inject", False)
        ),
        "--local-slots-data": getattr(args, "local_slots_data", None) is not None,
        "--mtvj-train-metric-head": bool(
            getattr(args, "mtvj_train_metric_head", False)
        ),
        "--mtvj-train-relation": bool(
            getattr(args, "mtvj_train_relation", False)
        ),
    }
    enabled_slots = [name for name, enabled in forbidden_slots.items() if enabled]
    if enabled_slots:
        raise ValueError(
            "--slot-free-policy forbids " + ", ".join(enabled_slots)
        )
    if getattr(args, "wmrm_only", False) and not getattr(args, "wmrm", False):
        raise ValueError("--wmrm-only requires --wmrm/--wam4va")
    if getattr(args, "va_only", False) and not getattr(args, "wmrm", False):
        raise ValueError("--va-only requires --wmrm/--wam4va")
    if getattr(args, "wmrm_only", False) and getattr(args, "va_only", False):
        raise ValueError("--world-only and --va-only are mutually exclusive")
    if getattr(args, "wmrm_only", False) and (
        args.head_only
        or args.servo_only
        or getattr(args, "action_vision_only", False)
    ):
        raise ValueError("--wmrm-only is mutually exclusive with --head-only/--servo-only/--action-vision-only")
    if getattr(args, "va_only", False) and (
        args.head_only
        or args.servo_only
        or getattr(args, "action_vision_only", False)
    ):
        raise ValueError(
            "--va-only is mutually exclusive with "
            "--head-only/--servo-only/--action-vision-only"
        )
    if getattr(args, "wmrm_only", False):
        # Stage-1 JEPA-style: freeze VA/Flow and optimize only World objectives.
        args.wmrm_adep_weight = 0.0
        args.mtvj_train_metric_head = False
        args.mtvj_train_relation = False
    if getattr(args, "wmrm", False) and float(getattr(args, "wmrm_world_weight", 1.0)) <= 0.0:
        raise ValueError("--wmrm requires positive --wmrm-world-weight")
    if getattr(args, "wmrm_world_weight", 1.0) < 0.0:
        raise ValueError("--wmrm-world-weight must be non-negative")
    static_constraint_weight = float(
        getattr(args, "wmrm_static_constraint_weight", 4.0)
    )
    if not math.isfinite(static_constraint_weight) or static_constraint_weight < 0.0:
        raise ValueError("--wmrm-static-constraint-weight must be finite and non-negative")
    if static_constraint_weight != 4.0 and not getattr(
        args, "visual_world_supervision", False
    ):
        raise ValueError(
            "--wmrm-static-constraint-weight only applies with --visual-world-supervision"
        )
    late_stage_anchor_weight = float(
        getattr(args, "wmrm_late_stage_anchor_weight", 0.0)
    )
    if not math.isfinite(late_stage_anchor_weight) or late_stage_anchor_weight < 0.0:
        raise ValueError(
            "--wmrm-late-stage-anchor-weight must be finite and non-negative"
        )
    if late_stage_anchor_weight != 0.0 and not getattr(
        args, "visual_world_supervision", False
    ):
        raise ValueError(
            "--wmrm-late-stage-anchor-weight only applies with "
            "--visual-world-supervision"
        )
    stage_overrides = visual_world_stage_weight_overrides(args)
    for index, flag in ((5, "--wmrm-stage-s5-weight"), (6, "--wmrm-stage-s6-weight")):
        if index not in stage_overrides:
            continue
        value = stage_overrides[index]
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{flag} must be finite and non-negative")
    if stage_overrides and not getattr(args, "visual_world_supervision", False):
        raise ValueError(
            "--wmrm-stage-s5-weight/--wmrm-stage-s6-weight only apply with "
            "--visual-world-supervision"
        )
    lr_wmrm_predictor = getattr(args, "lr_wmrm_predictor", None)
    if lr_wmrm_predictor is not None and (
        not math.isfinite(lr_wmrm_predictor) or lr_wmrm_predictor <= 0.0
    ):
        raise ValueError("--lr-wmrm-predictor must be a positive finite value")
    if lr_wmrm_predictor is not None and not getattr(args, "wmrm", False):
        raise ValueError("--lr-wmrm-predictor requires --wmrm/--wam4va")
    predictor_grad_clip = getattr(args, "wmrm_predictor_grad_clip", None)
    if predictor_grad_clip is not None and (
        not math.isfinite(predictor_grad_clip) or predictor_grad_clip <= 0.0
    ):
        raise ValueError("--wmrm-predictor-grad-clip must be a positive finite value")
    if predictor_grad_clip is not None and not getattr(args, "wmrm", False):
        raise ValueError("--wmrm-predictor-grad-clip requires --wmrm/--wam4va")
    if (
        lr_wmrm_predictor is not None or predictor_grad_clip is not None
    ) and getattr(args, "va_only", False):
        raise ValueError(
            "--lr-wmrm-predictor/--wmrm-predictor-grad-clip cannot be used with --va-only"
        )
    if (
        lr_wmrm_predictor is not None or predictor_grad_clip is not None
    ) and (
        args.head_only
        or args.servo_only
        or getattr(args, "action_vision_only", False)
    ):
        raise ValueError(
            "--lr-wmrm-predictor/--wmrm-predictor-grad-clip cannot be used with "
            "--head-only/--servo-only/--action-vision-only"
        )
    if int(getattr(args, "wmrm_cycle_steps", 6)) < 1:
        raise ValueError("--wmrm-cycle-steps must be >= 1")
    if getattr(args, "wmrm", False) and getattr(args, "wmrm_target", "dino") == "vjepa":
        dino_main = bool(getattr(args, "dino_main_vision", False))
        backbone = getattr(args, "main_vision_backbone", "vjepa")
        if dino_main or (backbone is not None and backbone != "vjepa"):
            raise ValueError(
                "wmrm_target=vjepa is incompatible with DINO main vision "
                "(dino_main_vision / main_vision_backbone != vjepa); "
                "do not infer V-JEPA from dense key 11"
            )
    if args.qwen_keep_layers < 0:
        raise ValueError("--qwen-keep-layers must be non-negative")
    if args.dino_qwen_cross_modal_bridge and args.dino_feature_cache is not None:
        raise ValueError(
            "--dino-qwen-cross-modal-bridge needs online DINO tail layers; "
            "the existing feature cache only contains blocks 11 and 23"
        )
    if args.dual_attention and args.sequential_coupling == 1:
        raise ValueError(
            "--dual-attention is incompatible with --sequential-coupling=1 "
            "(every VA layer is sequential; dual attention would never apply)"
        )
    if args.dual_attention and args.sequential_coupling > 1:
        print(
            "warning: --dual-attention only splits the non-sequential VA layers; "
            "sequential layers keep the legacy shared path"
        )
