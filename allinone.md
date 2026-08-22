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
