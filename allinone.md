# ORA0 本地结构相似论文检索报告

检索日期：2026-08-20  
检索范围：2024–2026  
API 检索式：`recurrent belief world action model state fusion action decoder future visual latent vision language action`  
判定对象：当前 active `peer_sync_h6` WAM4VA，而不是论文草稿中的 VA2 主线。

## 结论先行

没有找到与当前 WAM4VA 完全同构的论文。最接近的不是单独一篇，而是下列先例的交集：

- DUST：最像逐 block 的 action/vision 双流交互。
- Faster-WAM / ForeWAM：最像逐层 world/video K/V 进入 action transformer。
- EvoScene-VLA / RB-VLA：最像跨控制步的 action-updated recurrent scene/belief。
- DINO-WM / V-JEPA 2-AC：最像 action-conditioned frozen dense visual latent predictor。
- World Pilot：最像冻结 world prior、latent steering 与最终 flow action emitter。

本地仍未被单篇覆盖的组合是：**8-stage peer-synchronous proposal、独立 recurrent `belief + innovation + world_map`、`action -> world -> next VA layer` 的一拍延迟闭环、双向 stop-gradient，以及唯一可执行动作由最终 Flow head 发出。**

## 本地结构指纹

| 结构轴 | 当前实现 |
|---|---|
| 双流拓扑 | VA 与 WAM 维护独立状态；stage `i` 都读取 stage `i-1` 的 committed snapshot |
| 层间通信 | WAM message 作为下一 VA layer 的 attention K/V，而非直接修改动作 |
| World state | persistent `belief [8x512] + innovation [8x512] + world_map [1024x16x16]` |
| World target | executable H6 action-conditioned 下一决策完整 DINO dense map；future target stop-gradient |
| 梯度边界 | VA->WAM 与 WAM->VA 都 detach；world-only / VA-only 分相、分数据训练 |
| 动作输出 | WAM 内部有受监督 H6 readout 供 world condition，但唯一可执行输出是 Flow Matching head |

代码依据：[WAM 合同与状态](/home/ryan/Documents/robot/ORA0/va_compound/wmrm.py:1)、[peer-synchronous stage loop](/home/ryan/Documents/robot/ORA0/va_compound/model.py:2646)、[active runner](/home/ryan/Documents/robot/ORA0/scripts/run_mw_hard2_wam4va_visualmotion_peer_sync_h6_v1.sh:197)。旧 `wam.py` 已不再是当前实现。

注意：仓库论文主线 VA2 是另一条互斥结构线；其 future-latent predictor 被作者定义为 regularizer，不是完整 WM。`wmrm` 也与 `memory_split`、Direct/C2 controller 互斥，因此不能把 VA2、C2、WAM4VA 写成同一个已实现 checkpoint。

## 高信号结构近邻

| 论文 | 命中的本地结构 | 没命中的关键点 | 综合判断 |
|---|---|---|---|
| [DUST](https://arxiv.org/abs/2510.27607) | 每个 MMDiT block 保留 action/vision 双流，只在 shared attention 交换；独立噪声；action-conditioned forward-dynamics setting；flow matching | 无独立 recurrent WM state；同层 joint attention，不是下一层延迟 K/V；梯度联通；推理联合采样 future/action | **双流骨架最近** |
| [Faster-WAM](https://arxiv.org/abs/2608.04404) | aligned video/action stages；一次 video pass 形成逐层 K/V hierarchy；Action Transformer 在多个 stage 读取；flow action | 无 recurrent state；video hierarchy 不读 evolving action；joint training，无双向 detach；同层/区间注入 | **层级 K/V 拓扑最近** |
| [ForeWAM](https://arxiv.org/abs/2608.11605) | Video DiT 的 layer-wise Future-KV cache 供 Action DiT 全程读取；最终 executable action 来自 flow head | 单次 prefill、无 recurrent state；prefill 不读 policy action；action gradient 明确回传 video prefill；无 one-stage delay | **World-K/V 接口最近** |
| [EvoScene-VLA](https://arxiv.org/abs/2605.21862) | recurrent scene prefix 跨 action chunk；action decoder 同时输出动作与 scene update；新观测纠正 prior | 每 chunk 一次闭环，不是逐 layer peer；scene/action 同一 expert 且端到端；训练期 Scene Predictor 部署时删除 | **闭环状态语义最近** |
| [RB-VLA](https://arxiv.org/abs/2602.20659) | compact persistent action-conditioned belief；self-supervised world objectives；belief + intent 条件 diffusion policy | 串行 `belief estimator -> policy`；无 layer-wise K/V exchange；无 persistent dense map | **belief 语义最近** |
| [World Pilot](https://arxiv.org/abs/2606.12403) | 独立 WAM 的 scene-evolution latent 注入 VLA；trajectory prior 引导 flow action generator | 一次 perception-level steering；无 recurrent peer stages；不由当前 policy candidate action 条件化 | **冻结 WM steering 最近** |
| [Geometric Action Model](https://arxiv.org/abs/2606.17046) | 在 backbone 中间插 causal future predictor，预测 token 再穿过后半 backbone产生 future geometry/action | 单插入点、共享 backbone、无 recurrent belief、无 gradient isolation | **backbone 内插 predictor 最近** |
| [LaWAM](https://arxiv.org/abs/2606.15768) | 在视觉 foundation latent space 中做 action-conditioned future prediction，再用 latent visual subgoal 条件动作 | 单次 cascade；无 recurrent state、逐层 peer、delayed K/V | **latent subgoal 最近** |
| [DINO-WM](https://arxiv.org/abs/2411.04983) | frozen DINOv2 spatial patch target；action-conditioned next latent；自回归 rollout | 不是 VLA；没有 flow policy 或 world-token->policy K/V；动作由 MPC/CEM 求解 | **World predictor 半边最近** |

## 全量检索结果

### Semantic Scholar

| # | Title | Authors | Year | Citations | Venue | URL |
|---:|---|---|---:|---:|---|---|
| 1 | LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies | Jialei Chen, Kai Wang, Kanghao Chen, et al. | 2026 | 17 | arXiv.org | [Link](https://www.semanticscholar.org/paper/996f843abe83a075ec19dd66c7f15fe8e073380e) |
| 2 | EvoScene-VLA: Evolving Scene Beliefs Inside the Action Decoder for Chunked Robot Control | Chushan Zhang, Ruihan Lu, Jinguang Tong, et al. | 2026 | 2 | arXiv.org | [Link](https://www.semanticscholar.org/paper/1a8776d12f233047d9ba67e41d09bce90420926f) |
| 3 | Do World Action Models Generalize Better than VLAs? A Robustness Study | Zhanguang Zhang, Zhiyuan Li, Behnam Rahmati, et al. | 2026 | 15 | arXiv.org | [Link](https://www.semanticscholar.org/paper/ceedc67576d4dd95c8cc1fb0552f494c03281b29) |
| 4 | DIAL: Decoupling Intent and Action via Latent World Modeling for End-to-End VLA | Yi Chen, Yuying Ge, Hui Zhou, et al. | 2026 | 9 | arXiv.org | [Link](https://www.semanticscholar.org/paper/48bf030b6e8e23ec8cd0f5e64141ac567954d6ed) |
| 5 | PearlVLA: Progressive Embodied Action-Plan Refinement in Latent Space | Bochen Yang, Lianlei Shan | 2026 | 0 | arXiv.org | [Link](https://www.semanticscholar.org/paper/bba2cf431d516faf3fd94350b9ee2dced44d65ae) |
| 6 | Unifying Perception and Action: A Hybrid-Modality Pipeline with Implicit Visual Chain-of-Thought for Robotic Action Generation | Xiangkai Ma, Lekai Xing, Han Zhang, et al. | 2025 | 10 | arXiv.org | [Link](https://www.semanticscholar.org/paper/269422e60f757c1fe58ec4eb5e962dee080f43d1) |
| 7 | TFP: Temporally Conditioned Memory-Fusion Policies for Visuomotor Learning | Yushen Liang, Yue Peng, Baosheng Jin, et al. | 2026 | 0 | — | [Link](https://www.semanticscholar.org/paper/718ca18e7e0bd566f65bde7e652b3d17119109fd) |
| 8 | StreamVLA: Breaking the Reason-Act Cycle via Completion-State Gating | Tong Chen, Hang Wu, Jiasen Wang, et al. | 2026 | 1 | arXiv.org | [Link](https://www.semanticscholar.org/paper/17fcc2b1136f19af317101e8a22dbd1a5fec9f57) |

### OpenAlex

No results returned because the source request failed; see Source errors below.

### arXiv

| # | Title | Authors | Year | Citations | Venue | URL |
|---:|---|---|---:|---:|---|---|
| 1 | DiLA: Disentangled Latent Action World Models | Tianqiu Zhang, Muyang Lyu, Yufan Zhang, et al. | 2026 | 0 | arXiv | [Link](http://arxiv.org/abs/2605.15725v1) |
| 2 | Inference-Time Attention Steering for Vision-Language-Action Driving Models | Darshan Nagendra Prasad, Lars Ullrich, Knut Graichen | 2026 | 0 | arXiv | [Link](http://arxiv.org/abs/2608.17095v1) |
| 3 | Compositional Context Fine-Tuning Vision-Language Model for Complex Assembly Action Understanding from Videos | Hao Zheng, Jinyi Huang, Tiantian Zheng, et al. | 2026 | 0 | arXiv | [Link](http://arxiv.org/abs/2607.10797v1) |
| 4 | CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations | Anthony Liang, Pavel Czempin, Matthew M. Hong, et al. | 2025 | 0 | arXiv | [Link](http://arxiv.org/abs/2505.04999v2) |
| 5 | World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry | Yuejiang Liu, Fan Feng, Lingjing Kong, et al. | 2026 | 0 | arXiv | [Link](http://arxiv.org/abs/2604.01985v2) |
| 6 | Geometric Action Model for Robot Policy Learning | Jisang Han, Seonghu Jeon, Jaewoo Jung, et al. | 2026 | 0 | arXiv | [Link](http://arxiv.org/abs/2606.17046v2) |
| 7 | Unified Video Action Model | Shuang Li, Yihuai Gao, Dorsa Sadigh, et al. | 2025 | 0 | arXiv | [Link](http://arxiv.org/abs/2503.00200v3) |
| 8 | Your Vision-Language-Action Model Already Has Attention Heads For Path Deviation Detection | Jaehwan Jeong, Evelyn Zhu, Jinying Lin, et al. | 2026 | 0 | arXiv | [Link](http://arxiv.org/abs/2603.13782v1) |
| 9 | VLA-Thinker: Boosting Vision-Language-Action Models through Thinking-with-Image Reasoning | Chaoyang Wang, Wenrui Bao, Sicheng Gao, et al. | 2026 | 0 | arXiv | [Link](http://arxiv.org/abs/2603.14523v1) |
| 10 | EgoAction: Egocentric Action Composition with Reliability-Aware Temporal Fusion for the EPIC-KITCHENS Action Detection Challenge at CVPR 2026 | Zhiheng Fu, Zixu Li, Zhiwei Chen, et al. | 2026 | 0 | arXiv | [Link](http://arxiv.org/abs/2605.24496v2) |

### OpenReview

No results returned because the source request failed; see Source errors below.

### Crossref

| # | Title | Authors | Year | Citations | Venue | URL |
|---:|---|---|---:|---:|---|---|
| 1 | ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning | Chi-Pin Huang, Yueh-Hua Wu, Min-Hung Chen, et al. | 2025 | 0 | NeurIPS 38 | [DOI](https://doi.org/10.52202/085713-2773) |
| 2 | Vision-Language-Action and Vision Language Models for Robot Manipulation: A Comprehensive Review Towards Real-World Applications | Md Selim Sarowar, Sungho Kim | — | 0 | Preprint | [DOI](https://doi.org/10.20944/preprints202606.0400.v1) |
| 3 | The Dual-System Hierarchical Architecture: A Future Paradigm for Vision-Language-Action Models | Wenlong Chen, Zhen Tian, Zhou Zhou, et al. | 2025 | 0 | IEEE SWC 2025 | [DOI](https://doi.org/10.1109/swc65939.2025.00215) |
| 4 | PhysMargin-CoC: Provenance-Linked Visual-Analytic Evaluation of Predicate Exposure in Vision-Language-Action Models for Autonomous Driving | Jiwoo Jung | — | 0 | SSRN | [DOI](https://doi.org/10.2139/ssrn.7305959) |
| 5 | Looking to the future in language teacher action research | Anne Burns, Kenan Dikilitas | 2024 | 0 | Routledge Handbook | [DOI](https://doi.org/10.4324/9781003367352-39) |
| 6 | Vision-Language-Action Model for Electrical Power Operation Robots | Zhisong Zhang, Guozheng Peng, Peng Zhang, et al. | — | 0 | Authorea | [DOI](https://doi.org/10.22541/authorea.15004596/v1) |
| 7 | Scaling Vision-Language-Action Policy Adaptation via Action-Conditioned World Models | Yao Yeboah, Joseph Teye Ignatius Buertey, Kwabena Agyapong-Kodua, et al. | 2026 | 0 | ICMRE 2026 | [DOI](https://doi.org/10.1109/icmre69538.2026.11533985) |
| 8 | Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications | Kento Kawaharazuka, Jihoon Oh, Jun Yamada, et al. | — | 0 | TechRxiv | [DOI](https://doi.org/10.36227/techrxiv.175502755.53627529/v1) |
| 9 | Sensing the Action: Rethinking Sensor Modalities and Multi-Modal Fusion in Vision-Language-Action Models for Robotic Manipulation | Byoung Chul Ko | 2026 | 0 | Sensors | [DOI](https://doi.org/10.3390/s26113541) |
| 10 | Generating Robot Action Sequences: An Efficient Vision-Language Models with Visual Prompts | Weihao Cai, Yoshiki Mori, Nobutaka Shimada | 2024 | 3 | IWIS 2024 | [DOI](https://doi.org/10.1109/iwis62722.2024.10706068) |

### DBLP

No papers returned for this exact query.

### Model Knowledge（用原始论文页复核的高信号补充）

| # | Title | Year | Why included | URL |
|---:|---|---:|---|---|
| 1 | Dual-Stream Diffusion for World-Model Augmented VLA (DUST) | 2025/2026 | dual-stream、cross-modal block、flow-matching world/action | [arXiv](https://arxiv.org/abs/2510.27607) |
| 2 | Faster-WAM | 2026 | aligned stages、multi-depth K/V、sparse world-to-action interaction | [arXiv](https://arxiv.org/abs/2608.04404) |
| 3 | Foresight Without Seeing / ForeWAM | 2026 | layer-wise Future-KV cache -> Action DiT | [arXiv](https://arxiv.org/abs/2608.11605) |
| 4 | Recursive Belief Vision Language Action Models (RB-VLA) | 2026 | persistent action-conditioned belief + WM objective | [arXiv](https://arxiv.org/abs/2602.20659) |
| 5 | World Pilot | 2026 | scene latent and trajectory prior steering a VLA/flow action generator | [arXiv](https://arxiv.org/abs/2606.12403) |
| 6 | Bridge-WA | 2026 | compact future/change/flow priors as attention memories | [arXiv](https://arxiv.org/abs/2607.02195) |
| 7 | LaWAM | 2026 | action-conditioned visual-foundation-model latent subgoal | [arXiv](https://arxiv.org/abs/2606.15768) |
| 8 | Light-WAM | 2026 | multi-layer state fusion into direct action prediction | [arXiv](https://arxiv.org/abs/2606.08242) |
| 9 | DINO-WM | 2024 | action-conditioned frozen DINOv2 patch dynamics | [arXiv](https://arxiv.org/abs/2411.04983) |
| 10 | V-JEPA 2 / V-JEPA 2-AC | 2025 | action-conditioned dense latent world predictor ancestor | [arXiv](https://arxiv.org/abs/2506.09985) |

## Source errors

```text
[openreview] Error: {'name': 'IncompleteRegistrationError', 'message': 'Your profile could not be activated and more information is required, please click on "Didn\'t receive email confirmation?" to receive a new confirmation link to edit your profile. (2026-08-20-6371346)', 'status': 400, 'details': {'reqId': '2026-08-20-6371346'}}
[open_alex] Error: 504 Server Error: Gateway Timeout for url: https://api.openalex.org/works?search.semantic=recurrent+belief+world+action+model+state+fusion+action+decoder+future+visual+latent+vision+language+action&filter=publication_year%3A2024-2026&sort=relevance_score%3Adesc&page=1&per-page=10
```

API search returned 28 papers from six configured sources; OpenAlex and OpenReview returned zero because of the errors above, and DBLP returned zero matches.

## Overview

The exact-signature search does not reveal a single prior work with all of WAM4VA's mechanisms. Recent work has, however, occupied nearly every individual axis: dual-stream world/action diffusion (DUST), layer-wise K/V transfer into action denoisers (Faster-WAM and ForeWAM), recurrent action-updated belief (EvoScene-VLA and RB-VLA), and frozen dense visual-latent dynamics (DINO-WM and V-JEPA 2-AC). The defensible claim is therefore a novel **composition and interface**, not the first VLA+WM, first recurrent state, or first world-to-action K/V design.

## Trends

The field is moving from expensive decoded future video toward compact latent foresight, from auxiliary world losses toward inference-time future conditioning, and from monolithic joint generation toward explicit world-to-action interfaces. The August 2026 Faster-WAM and ForeWAM papers make layer-wise K/V conditioning especially important prior art for this project.

## Key themes

The dominant themes are latent rather than RGB future prediction; world state used as action context; flow/diffusion action chunks; persistent state for partial observability; and efficiency through cached or sparse cross-stream interaction.

## Keywords frequency (top 5)

Token counts over the 28 API-returned titles, with `model/models/modeling` merged and hyphenated terms split:

| Keyword | Count |
|---|---:|
| action | 30 |
| model | 20 |
| language | 15 |
| vision | 14 |
| world | 9 |

## Most cited accepted paper

Among API-returned entries with an explicit formal venue, **Generating Robot Action Sequences: An Efficient Vision-Language Models with Visual Prompts** (IWIS 2024) has the highest reported citation count: 3. Citation counts are source snapshots and should not be treated as stable.

## Most cited first author

Within the returned records, **Jialei Chen** leads through LaWAM with 17 Semantic Scholar citations. This is paper-level citation evidence, not an author-total citation metric.

## Recommendations for reading

Read in this order: **DUST -> Faster-WAM -> ForeWAM -> EvoScene-VLA -> RB-VLA**. This sequence isolates the five parts most relevant to WAM4VA: dual streams, stage-wise K/V, hidden future cache, recurrent scene state, and predictive belief. Then use DINO-WM or V-JEPA 2-AC only to audit the dense DINO-world-predictor half.

---

# Meta-World 验收协议与 seed 核验

检索日期：2026-08-27

检索范围：2024–2026

API 检索式：`EVO-1 FabriVLA SUREFlow Meta-World robot learning evaluation seeds`

## 结论先行

- Evo-1：每任务 10 次 episode；单次官方脚本用 reset seed `4042..4051`；论文结果再对 5 次独立运行取平均。
- FabriVLA v3：每任务 10 次 episode；当前官方脚本用 reset seed `4048..4057`。未报告 5 次独立运行；旧 v1 文本曾写 base seed 4042，复现时必须固定版本。
- SUREFlow：论文只明确每任务 50 rollouts，没有公开说明它们是否对应 50 个唯一 seed，也没有报告独立训练/评估 seed 数。
- EvoMind 榜单口径是 50 个任务和四个难度桶；仅看总 episode 成功率不足以复现榜单分数。

## 直接证据

| 工作 | 任务数 | 每任务 rollout | 可核验的 episode seed | 独立运行 | episode horizon | 汇总口径 |
|---|---:|---:|---|---:|---:|---|
| Evo-1 | 50 | 10 | 官方脚本 `4042 + episode` | 论文写 5 次 | 400 | 四个难度桶等权平均 |
| FabriVLA v3 | 50 | 10 | 当前脚本 `4048 + episode` | 未报告多次 | 400 | 四个难度桶等权平均 |
| SUREFlow | 50 | 50 | 未报告 | 未报告 | 未找到明确值 | 表格采用四难度桶平均 |

主要原始来源：[Evo-1 论文](https://arxiv.org/html/2511.04555v2)、[Evo-1 官方评估代码](https://github.com/MINT-SJTU/Evo-1/blob/evo1-flash/MetaWorld_evaluation/mt50_evo1_client_prompt.py)、[FabriVLA v3 论文](https://arxiv.org/html/2607.08575v3)、[FabriVLA 官方评估代码](https://github.com/Youi-FabriX/FabriVLA/blob/main/evaluations/metaworld/evaluate_mt50.py)、[SUREFlow 论文](https://arxiv.org/html/2607.10504v1)、[EvoMind Meta-World 榜单](https://studio.evomind-tech.com/benchmarks/metaworld?sources=paper%2Cstudio)。

## Semantic Scholar

本次精确检索返回 0 篇。

## OpenAlex

| # | Title | Authors | Year | Citations | Venue | URL |
|---:|---|---|---:|---:|---|---|
| 1 | A Tutorial on Meta-Reinforcement Learning | Jacob Beck; Risto Vuorio; Evan Zheran Liu; et al. | 2025 | 42 | Foundations and Trends in Machine Learning | [DOI](https://doi.org/10.1561/2200000080) |
| 2 | Autonomous Landing of the Quadrotor on the Mobile Platform via Meta Reinforcement Learning | Qianqian Cao; Ziyi Liu; Hai Yu; et al. | 2024 | 21 | IEEE T-ASE | [DOI](https://doi.org/10.1109/tase.2024.3377810) |
| 3 | Grow Your Limits: Continuous Improvement with Real-World RL for Robotic Locomotion | Laura Smith; Yun-Hao Cao; Sergey Levine | 2024 | 12 | ICRA | [DOI](https://doi.org/10.1109/icra57147.2024.10610485) |
| 4 | MAGIC VFM: Meta-Learning Adaptation for Ground Interaction Control With Visual Foundation Models | Elena Sorina Lupu; Fengze Xie; James A. Preiss; et al. | 2024 | 11 | IEEE Transactions on Robotics | [DOI](https://doi.org/10.1109/tro.2024.3475212) |
| 5 | CRiSE 1-3-7: Compilations of Real-World inspired Robotic Task Simulation Environments for Meta-RL | Johannes Ivancsics | 2024 | 0 | TU Wien repository | [DOI](https://doi.org/10.34726/hss.2024.79588) |
| 6 | METAVerse: Meta-Learning Traversability Cost Map for Off-Road Navigation | Junwon Seo; Taekyung Kim; Seongyong Ahn; et al. | 2024 | 7 | IROS | [DOI](https://doi.org/10.1109/iros58592.2024.10802444) |
| 7 | RoboVerse: A Unified Platform, Benchmark and Dataset for Scalable and Generalizable Robot Learning | Haoran Geng; Feishi Wang; Songlin Wei; et al. | 2025 | 7 | RSS | [DOI](https://doi.org/10.15607/rss.2025.xxi.022) |
| 8 | TEAM: Task-Clustering and Enhanced Adaptability for Meta-Reinforcement Learning on Robotics Through Multi-Task Diffusion and Optimization | Joshua W. K. Ho; Chien-Min Wang; Chung-Ta King; et al. | 2024 | 0 | IRC | [DOI](https://doi.org/10.1109/irc63610.2024.00024) |
| 9 | TaCoD: Tasks-Commonality-Aware World in Meta Reinforcement Learning | Xuantang Xiong; Shuang Xu; Bo Xu | 2024 | 2 | IJCNN | [DOI](https://doi.org/10.1109/ijcnn60899.2024.10649914) |
| 10 | Meta-Learning for Robotic Vision Applications | Ning Gao | 2025 | 0 | KITopen | [DOI](https://doi.org/10.5445/ir/1000180168) |

## arXiv

| # | Title | Authors | Year | Citations | Venue | URL |
|---:|---|---|---:|---:|---|---|
| 1 | Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment | Tao Lin; Yilei Zhong; Yuxin Du; et al. | 2025 | 0 | arXiv | [2511.04555v2](https://arxiv.org/abs/2511.04555v2) |
| 2 | Meta-DT: Offline Meta-RL as Conditional Sequence Modeling with World Model Disentanglement | Zhi Wang; Li Zhang; Wenhao Wu; et al. | 2024 | 0 | arXiv | [2410.11448v2](https://arxiv.org/abs/2410.11448v2) |
| 3 | Robot Learning: A Tutorial | Francesco Capuano; Caroline Pascal; Adil Zouitine; et al. | 2025 | 0 | arXiv | [2510.12403v1](https://arxiv.org/abs/2510.12403v1) |
| 4 | State-of-the-art in Robot Learning for Multi-Robot Collaboration: A Comprehensive Survey | Bin Wu; C. Steve Suh | 2024 | 0 | arXiv | [2408.11822v1](https://arxiv.org/abs/2408.11822v1) |
| 5 | Robot Policy Evaluation for Sim-to-Real Transfer: A Benchmarking Perspective | Xuning Yang; Clemens Eppner; Jonathan Tremblay; et al. | 2025 | 0 | arXiv | [2508.11117v1](https://arxiv.org/abs/2508.11117v1) |
| 6 | Robot Trains Robot: Automatic Real-World Policy Adaptation and Learning for Humanoids | Kaizhe Hu; Haochen Shi; Yao He; et al. | 2025 | 0 | arXiv | [2508.12252v2](https://arxiv.org/abs/2508.12252v2) |
| 7 | SUREFlow: State-space Uncertainty-aware REsidual Flow Matching for Robust Robot Manipulation | Md Tanvir Islam; Sai Navaneet Peddapalli; Sangmoon Lee; et al. | 2026 | 0 | arXiv | [2607.10504v1](https://arxiv.org/abs/2607.10504v1) |
| 8 | Robot Learning from Human Videos: A Survey | Junyi Ma; Erhang Zhang; Haoran Yang; et al. | 2026 | 0 | arXiv | [2604.27621v1](https://arxiv.org/abs/2604.27621v1) |
| 9 | The Sound of Simulation: Learning Multimodal Sim-to-Real Robot Policies with Generative Audio | Renhao Wang; Haoran Geng; Tingle Li; et al. | 2025 | 0 | arXiv | [2507.02864v2](https://arxiv.org/abs/2507.02864v2) |
| 10 | ClutterGen: A Cluttered Scene Generator for Robot Learning | Yinsen Jia; Boyuan Chen | 2024 | 0 | arXiv | [2407.05425v2](https://arxiv.org/abs/2407.05425v2) |

## OpenReview

本次返回 0 篇；请求失败，见 Source errors。

## Crossref

| # | Title | Authors | Year | Citations | Venue | URL |
|---:|---|---|---:|---:|---|---|
| 1 | The Human-Robot Interactive Reinforcement Learning for Robot Navigation of The Factory Transportation System in Grid World Environment | Xumin Gao | — | 0 | — | [DOI](https://doi.org/10.36227/techrxiv.176186638.80043394/v1) |
| 2 | Development and Evaluation of a Deep Q-Network-Based Robot Learning Paradigm in Real-World Human-Robot Collaborative Tasks | Garrett Modery; Weitian Wang; Rui Li; et al. | 2025 | 0 | CASE | [DOI](https://doi.org/10.1109/case58245.2025.11164081) |
| 3 | RoboMorph: In-Context Meta-Learning for Robot Dynamics Modeling | Manuel Bazzi; Asad Shahid; Christopher Agia; et al. | 2024 | 2 | ICINCO | [DOI](https://doi.org/10.5220/0012945500003822) |
| 4 | Meta-Learning for Dynamic Multi-Robot Task Scheduling | Peng Song; Huaiyu Chen; Kaixin Cui; et al. | — | 1 | SSRN | [DOI](https://doi.org/10.2139/ssrn.5044505) |
| 5 | Learning in World Bank Lending: An Independent Evaluation | — | 2025 | 0 | — | [DOI](https://doi.org/10.1596/ieg197067) |
| 6 | Cosmos-Surg-DVRK: World Foundation Model-Based Automated Online Evaluation of Surgical Robot Policy Learning | Lukas Zbinden; Nigel Nelson; Juo-Tung Chen; et al. | 2026 | 1 | IEEE RA-L | [DOI](https://doi.org/10.1109/lra.2026.3675962) |
| 7 | Coordinated World Model Learning for Deep Space Robot Teams | Andrzej M.J. Skulimowski | 2026 | 1 | IEEE Aerospace Conference | [DOI](https://doi.org/10.1109/aero66936.2026.11519895) |
| 8 | Transperitoneal versus retroperitoneal robot-assisted partial nephrectomy: a systematic review and meta-analysis | Nikita Shrivastava; Priyank Bhargava; Gopal Sharma; et al. | 2024 | 24 | World Journal of Urology | [DOI](https://doi.org/10.1007/s00345-024-04796-7) |
| 9 | Gaining Python Skills Through Interactive Education Robot Ozobot EVO | Maya Staikova | 2025 | 0 | TechSys 2025 | [DOI](https://doi.org/10.3390/engproc2025100015) |
| 10 | Editorial: Reinforcement learning for real-world robot navigation | Pengqin Wang; Xiaocong Li; Meixin Zhu; et al. | 2026 | 0 | Frontiers in Robotics and AI | [DOI](https://doi.org/10.3389/frobt.2026.1861947) |

## DBLP

本次精确检索返回 0 篇。

## Model Knowledge（经论文页与官方仓库复核的补充）

| # | Title | Authors | Year | Citations | Venue | URL |
|---:|---|---|---:|---:|---|---|
| 1 | FabriVLA: Learning Efficient Vision-Language-Action Model with Fine-Grained Cross-Modal Fabric | FabriX team | 2026 | — | arXiv | [2607.08575v3](https://arxiv.org/abs/2607.08575v3) |

## Source errors

```text
[openreview] Error: {'name': 'IncompleteRegistrationError', 'message': 'Your profile could not be activated and more information is required, please click on "Didn\'t receive email confirmation?" to receive a new confirmation link to edit your profile. (2026-08-26-7016065)', 'status': 400, 'details': {'reqId': '2026-08-26-7016065'}}
```

API search returned 30 records from the six configured sources. Only Evo-1 and SUREFlow directly matched the target protocol question; FabriVLA was added from its verified primary paper and repository.

## Overview

The protocol evidence is unusually uneven. Evo-1 reports both ten episodes per task and five independent runs. FabriVLA reports ten episodes per task and exposes the episode reset sequence in code, but no multi-run average. SUREFlow reports fifty rollouts per task without disclosing a seed schedule. Therefore rollout count, episode-reset seed count, and independent model/run count must remain separate fields.

## Trends

Recent Meta-World VLA work is converging on all 50 tasks, a 400-step cap, and four difficulty buckets, but statistical reporting is not standardized: papers mix single-checkpoint rollouts, repeated evaluation runs, and independently trained runs.

## Key themes

The relevant themes are deterministic reset schedules, independent-run averaging, task-balanced versus tier-balanced metrics, fixed episode horizon, and raw per-trial trace retention.

## Keywords frequency (top 5)

Approximate title-level counts over the 30 API records, merging singular/plural, hyphenation, and common morphology:

| Keyword family | Count |
|---|---:|
| robot / robotic / robotics | 26 |
| learning / RL | 23 |
| meta | 14 |
| world / real-world | 12 |
| evaluation | 4 |

## Most cited accepted paper

Among the API-returned formal publications, **A Tutorial on Meta-Reinforcement Learning** has the highest reported snapshot citation count (42).

## Most cited first author

Within this result set, **Jacob Beck** leads via the same paper with 42 citations. This is a returned-paper count, not a complete author-level citation profile.

## Recommendations for reading

Read only the three primary protocol artifacts needed for acceptance: Evo-1 paper plus evaluator, FabriVLA v3 paper plus evaluator, and SUREFlow section 4.1. The other exact-query records are background or false-positive matches and should not influence the ORA0 acceptance threshold.

---

# DAgger、纠错干预与恢复轨迹论文检索报告

检索日期：2026-08-27

检索范围：2010–2026
检索式：`robot imitation learning DAgger corrective interventions recovery demonstrations`

## Semantic Scholar（10 篇）

| # | Title | Date | Venue | Citations |
|---:|---|---:|---|---:|
| [1](https://www.semanticscholar.org/paper/ca275d9ef374dbbf81a6f6175cac2311526308c3) | IntervenGen: Interventional Data Generation for Robust and Data-Efficient Robot Imitation Learning | 2024 | IROS | 38 |
| [2](https://www.semanticscholar.org/paper/d42f55af51fc631f3a0637af966b52c698693626) | Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism | 2025 | ICML | 6 |
| [3](https://www.semanticscholar.org/paper/22c099753403e7b56739aee366bf5b46d553c144) | RaC: Robot Learning for Long-Horizon Tasks by Scaling Recovery and Correction | 2025 | IEEE Transactions on Robotics | 38 |
| [4](https://www.semanticscholar.org/paper/94fb66d93292570425efad07878cf25a4a80bfbf) | WM-DAgger: Enabling Efficient Data Aggregation for Imitation Learning with World Models | 2026 | arXiv | 5 |
| [5](https://www.semanticscholar.org/paper/7746cdc32f3d7e37b631c139b786b79b3c75d6d1) | Beyond Monotonic Progress: Retry-Supervised Value Learning for Robot Imitation | 2026 | arXiv | 0 |
| [6](https://www.semanticscholar.org/paper/443d4d2bb17a39cc535cd205c7cc47f754f1cd48) | Human-Robot Copilot for Data-Efficient Imitation Learning | 2026 | arXiv | 0 |
| [7](https://www.semanticscholar.org/paper/b7c107fa7b2ab017439858a9cc2d7a5ca15c320b) | Efficient Active Imitation Learning with Random Network Distillation | 2024 | ICLR | 9 |
| [8](https://www.semanticscholar.org/paper/d0d57bd76e5e9f3038d34798fdf4198f2aafc7cf) | AutoIntervene: Calibrated Intervention for Action-Chunking Imitation Learning Policies | 2026 | — | 1 |
| [9](https://www.semanticscholar.org/paper/e8952ebc3ba4678d8c6192eb8c3c51ee032258d0) | HACTS: a Human-As-Copilot Teleoperation System for Robot Learning | 2025 | IROS | 10 |
| [10](https://www.semanticscholar.org/paper/0aa7ed66a12f55f96d114210d7219c1cae0325f6) | Online Imitation Learning for Manipulation via Decaying Relative Correction through Teleoperation | 2025 | IROS | 4 |

## OpenAlex（10 篇）

| # | Title | Date | Venue | Citations |
|---:|---|---:|---|---:|
| [1](https://openalex.org/W2989897153) | Causal Confusion in Imitation Learning | 2019 | UvA-DARE | 63 |
| [2](https://doi.org/10.1109/cvpr52729.2023.01717) | NeRF in the Palm of Your Hand: Corrective Augmentation for Robotics via Novel-View Synthesis | 2023 | CVPR | 42 |
| [3](https://doi.org/10.1177/0278364919871998) | Reinforcement learning of motor skills using Policy Search and human corrective advice | 2019 | IJRR | 23 |
| [4](https://openalex.org/W3031773556) | On-Policy Robot Imitation Learning from a Converging Supervisor | 2019 | CoRL | 14 |
| [5](https://doi.org/10.48550/arxiv.2012.06733) | Human-in-the-Loop Imitation Learning using Remote Teleoperation | 2020 | arXiv | 10 |
| [6](https://doi.org/10.1109/icra.2011.5979792) | Physical human robot interaction in imitation learning | 2011 | ICRA | 10 |
| [7](https://doi.org/10.48550/arxiv.2509.07953) | RaC: Robot Learning for Long-Horizon Tasks by Scaling Recovery and Correction | 2025 | arXiv | 1 |
| [8](https://doi.org/10.1007/s10846-021-01312-6) | Endowing Robots with Longer-term Autonomy by Recovering from External Disturbances in Manipulation Through Grounded Anomaly Classification and Recovery Policies | 2021 | Journal of Intelligent & Robotic Systems | 9 |
| [9](https://openalex.org/W3079699067) | Residual Learning from Demonstration | 2020 | arXiv | 5 |
| [10](https://doi.org/10.48550/arxiv.2503.15368) | Online Imitation Learning for Manipulation via Decaying Relative Correction through Teleoperation | 2025 | arXiv | 2 |

## arXiv（10 篇）

| # | Title | Date | Venue | Citations |
|---:|---|---:|---|---:|
| [1](http://arxiv.org/abs/1907.03423v7) | On-Policy Robot Imitation Learning from a Converging Supervisor | 2019 | arXiv | 0 |
| [2](http://arxiv.org/abs/2310.14196v1) | Learning to Discern: Imitating Heterogeneous Human Demonstrations with Preference and Representation Learning | 2023 | arXiv | 0 |
| [3](http://arxiv.org/abs/1907.03976v3) | Better-than-Demonstrator Imitation Learning via Automatically-Ranked Demonstrations | 2019 | arXiv | 0 |
| [4](http://arxiv.org/abs/2105.06411v2) | Coarse-to-Fine Imitation Learning: Robot Manipulation from a Single Demonstration | 2021 | arXiv | 0 |
| [5](http://arxiv.org/abs/1711.10137v2) | One-Shot Reinforcement Learning for Robot Navigation with Interactive Replay | 2017 | arXiv | 0 |
| [6](http://arxiv.org/abs/2402.17768v2) | Diffusion Meets DAgger: Supercharging Eye-in-hand Imitation Learning | 2024 | arXiv | 0 |
| [7](http://arxiv.org/abs/1709.04905v1) | One-Shot Visual Imitation Learning via Meta-Learning | 2017 | arXiv | 0 |
| [8](http://arxiv.org/abs/2310.17596v1) | MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations | 2023 | arXiv | 0 |
| [9](http://arxiv.org/abs/2008.00524v2) | Interactive Imitation Learning in State-Space | 2020 | arXiv | 0 |
| [10](http://arxiv.org/abs/2303.01497v1) | Teach a Robot to FISH: Versatile Imitation from One Minute of Demonstrations | 2023 | arXiv | 0 |

## OpenReview（0 篇）

检索错误（原样保留）：`IncompleteRegistrationError: Your profile could not be activated and more information is required, please click on "Didn't receive email confirmation?" to receive a new confirmation link to edit your profile. (2026-08-27-5100674)`

## Crossref（10 篇）

| # | Title | Date | Venue | Citations |
|---:|---|---:|---|---:|
| [1](https://doi.org/10.1109/lra.2025.3536297) | Greedy-DAgger - A Student Rollout Efficient Imitation Learning Algorithm | 2025 | IEEE RA-L | 6 |
| [2](https://doi.org/10.1109/hri.2019.8673287) | Learning from Corrective Demonstrations | 2019 | HRI | 5 |
| [3](https://doi.org/10.1109/lra.2022.3196122) | Learning Category-Level Generalizable Object Manipulation Policy Via Generative Adversarial Self-Imitation Learning From Demonstrations | 2022 | IEEE RA-L | 29 |
| [4](https://doi.org/10.1109/icra40945.2020.9196602) | Zero-shot Imitation Learning from Demonstrations for Legged Robot Visual Navigation | 2020 | ICRA | 15 |
| [5](https://doi.org/10.15607/rss.2024.xx.048) | Diffusion Meets DAgger: Supercharging Eye-in-hand Imitation Learning | 2024 | RSS | 11 |
| [6](https://doi.org/10.1109/access.2023.3325194) | Imitation Learning for Agnostic Battery Charging: A DAGGER-Based Approach | 2023 | IEEE Access | 14 |
| [7](https://doi.org/10.2139/ssrn.5719301) | Imitation Learning from Diverse Suboptimal Demonstrations | — | SSRN | 0 |
| [8](https://doi.org/10.15607/rss.2023.xix.009) | Teach a Robot to FISH: Versatile Imitation from One Minute of Demonstrations | 2023 | RSS | 32 |
| [9](https://doi.org/10.1016/j.ins.2022.04.015) | Best-in-class imitation: Non-negative positive-unlabeled imitation learning from imperfect demonstrations | 2022 | Information Sciences | 6 |
| [10](https://doi.org/10.1109/icra.2019.8793698) | HG-DAgger: Interactive Imitation Learning with Human Experts | 2019 | ICRA | 137 |

## DBLP（0 篇）

检索错误（原样保留）：`503 Server Error: Service Unavailable for url: https://dblp.org/search/publ/api?q=robot+imitation+learning+DAgger+corrective+interventions+recovery+demonstrations&format=json&h=10&f=0`

## Model Knowledge（1 篇，经原始论文页核验）

| # | Title | Date | Venue | Notes |
|---:|---|---:|---|---|
| [1](https://proceedings.mlr.press/v15/ross11a.html) | A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning | 2011 | AISTATS | DAgger 原始论文；API 精确查询漏掉的奠基工作 |

## Overview

本次以 2010–2026 为窗口，从六个 API 源取得 40 条记录；OpenReview 因账户注册错误、DBLP 因 503 未返回结果。记录中存在跨源重复，核心谱系从 2011 年 DAgger 的 learner-induced state aggregation，发展到 DART/HG-DAgger 的扰动与人工接管，再到 2024–2026 年面向 action chunk、生成式数据和显式 recovery/correction 的方法。

## Trends

2011–2020 年的工作主要解决行为克隆的 covariate shift 与人工在线标注负担；2023 年后研究重点转向更低成本地合成或扩增干预数据；2024–2026 年则明显聚焦长时程操作、动作块策略、自动干预门控、world model 辅助聚合，以及明确的 retry/recovery 行为。主要正式发表 venue 包括 AISTATS、CoRL、ICRA、IROS、RSS、ICLR 与 ICML。

## Key themes

1. **On-policy aggregation**：执行当前策略，在它诱导的状态上请求专家动作；代表作 DAgger、SHIV。
2. **Corrective intervention**：只在风险或失败将要发生时接管，减少专家负担；代表作 HG-DAgger、IntervenGen。
3. **Perturb-and-recover**：给专家轨迹注入扰动并学习恢复；代表作 DART。
4. **Generated recovery data**：用扩散或场景生成覆盖 OOD 状态；代表作 Diffusion Meets DAgger、IntervenGen。
5. **Explicit retry/recovery**：将恢复与纠正片段作为后训练阶段；代表作 RaC、RACER、AutoIntervene。

## Keywords frequency

对标题与摘要做词形合并后的近似计数：

| Keyword | Count |
|---|---:|
| imitation learning | 29 |
| robot / robotics | 24 |
| demonstration | 16 |
| correction / intervention / recovery | 15 |
| DAgger / data aggregation | 9 |

## Most cited by accepted paper

跨源同名论文去重并取最高引用快照：

| Rank | Title | Year | Citations |
|---:|---|---:|---:|
| 1 | HG-DAgger: Interactive Imitation Learning with Human Experts | 2019 | 137 |
| 2 | Causal Confusion in Imitation Learning | 2019 | 63 |
| 3 | NeRF in the Palm of Your Hand | 2023 | 42 |
| 4 | IntervenGen | 2024 | 38 |
| 5 | RaC | 2025 | 38 |

## Most cited by first author

按本次去重结果中的最高引用快照统计：

| Rank | Author | Papers in set | Total citations |
|---:|---|---:|---:|
| 1 | Michael Kelly | 1 | 137 |
| 2 | Pim de Haan | 1 | 63 |
| 3 | Allan Zhou | 1 | 42 |
| 4 | Ryan Hoque | 1 | 38 |
| 5 | Zheyuan Hu | 1 | 38 |

## Recommendations for reading

1. [DAgger（AISTATS 2011）](https://proceedings.mlr.press/v15/ross11a.html)：理解为什么必须在学习器自己诱导的状态分布上训练。
2. [DART（CoRL 2017）](https://proceedings.mlr.press/v78/laskey17a.html)：不完整跑 DAgger，而用扰动迫使专家演示恢复；与 MetaWorld scripted expert 很匹配。
3. [HG-DAgger（ICRA 2019）](https://arxiv.org/abs/1810.02890)：仅在高风险状态由专家接管。
4. [Diffusion Meets DAgger（RSS 2024）](https://www.roboticsproceedings.org/rss20/p048.html)：以生成方式扩充跑偏状态，直接针对视觉操作的误差累积。
5. [RaC（2025）](https://arxiv.org/abs/2509.07953)：与 ORA0 当前问题最接近；预训练后增加 recovery/correction 后训练阶段。
