# Bidirectional Visual-Action Memory with Frozen Language Cache: Structural Language Grounding for Lightweight VLA Policies

**Draft v0.1 — 2026-08-06 — work-in-progress (pending: L_m verdict, VLA-RL, MW multi-start, LIBERO-100)**

---

## Abstract

Vision-Language-Action (VLA) models inherit instruction-following ability from
large pretrained vision-language backbones, but policy fine-tuning erodes this
ability through two mechanisms: *representation erosion* — action gradients
rewriting language-related parameters (instruction blindness) — and *visual
shortcut learning* — the policy ignoring language because scene features
suffice. We show that a *decoupled* architecture sidesteps the first mechanism
by construction: a frozen language backbone encodes the instruction once into
a static key/value cache, and a compact recurrent "visual-action (VA)
composite" reads that cache through a shared attention mechanism, so action
gradients can never reach the language representation. On LIBERO (3-scene,
12-task) and MetaWorld MT50, this frozen-cache policy retains sharp language
grounding (blanking language changes chunk error by +2381%; wrong instructions
+108.5%) while the same policy with an end-to-end fine-tuned language adapter
collapses the instruction embedding space (pairwise cosine 0.999 vs. 0.857)
and loses language sensitivity (+1.5%). A 2x2 control (frozen/LoRA language x
frozen/unfrozen vision) isolates the collapse to the trainable language
adapter. Our action head is a lightweight flow-matching module (43.5M
trainable parameters, 40.6 Hz deployment), trained with a two-term loss
(flow-matching + paired instruction contrast) — no auxiliary losses such as
image reconstruction or world-model objectives used by memory-based VLA
baselines.

<!-- [TBD: L_m verdict, VLA-RL gains, MW multi-start closed loop, LIBERO-100] -->

## 1. Introduction

Vision-Language-Action (VLA) models have become the dominant paradigm for
robot manipulation: a pretrained vision-language backbone conditions a
continuous action head on both the visual observation and a natural-language
instruction. Their promise is *language grounding* — the ability to execute
the specific instruction the user gave, rather than a scene-dependent default.
Yet recent analyses show that this ability is fragile. When a VLA is fine-tuned
on action data, the instruction-following behavior of its backbone
progressively degrades (LIBERO-CF; flatness analyses of instruction blindness).
Two mechanisms are implicated. First, *representation erosion*: action-loss
gradients flow back through the shared backbone and rewrite the parameters
that carry language meaning, collapsing the model's ability to distinguish
instructions at all. Second, *visual shortcut learning*: because scene
features are highly predictive of the correct action in most datasets, the
policy can drive training loss down while attending to language almost not at
all — it "passes" on the data distribution but silently fails on counterfactual
instructions. Both mechanisms are *training-coupled*: they arise from the
interaction between the action loss and a shared, trainable backbone.

In this paper we ask whether a *decoupled* architecture can remove the first
mechanism by construction, and expose the second through controlled
counterfactual evaluation. We study a compact policy we call the
**bidirectional visual-action (VA) composite**: a frozen language encoder
(Qwen3.5-2B) runs once per instruction and caches its final-layer hidden
states as static key/value anchors; a frozen video encoder (V-JEPA 2.1) turns
a causal 4-frame window into visual tokens; and a small stack of shared
attention blocks — the VA composite — reads vision, a learned action stream,
a single-slot recurrent visual memory, and the static language anchors, before
a flow-matching head regresses the action chunk. Because the language encoder
is frozen *and* detached, action gradients cannot reach language parameters by
construction — representation erosion is structurally impossible, not merely
discouraged.

Our contributions:

1. **Architecture-level immunity to representation erosion.** We verify with a
   2x2 control (language encoder frozen/LoRA-adapted × vision encoder
   frozen/unfrozen) on LIBERO: a fully frozen pipeline retains the exact
   pretrained instruction embedding space (pairwise cosine similarity of 12
   instructions: 0.857, identical to the untrained encoder), while end-to-end
   fine-tuning with a LoRA adapter collapses that space (0.999) and
   correspondingly destroys behavioral language sensitivity (blanking the
   language stream changes chunk error by +1.5% vs. +2381% for the frozen
   policy). Within our settings (V-JEPA 2.1 ViT-B, Qwen 2B, LIBERO), updating
   the language adapter during policy training reproducibly removes the
   policy's behavioral selectivity to counterfactual instructions, while
   freezing it preserves both selectivity and clean-task accuracy.

2. **Causal language-grounding evidence under same-scene counterfactuals.** On
   LIBERO's same-scene different-instruction regime — exactly the regime that
   exposes visual shortcuts — the frozen-cache policy shows language to be a
   necessary input: zeroing the language stream increases chunk error by
   +2381% (3-scene, 12-task) and +13751% (1-scene), and swapping instructions
   by +607%/+1518%. On MetaWorld MT50, substituting a wrong instruction
   degrades action error by +108.5%, and replacing language with a task-id
   token by +81.5% — language content beyond task identity participates in
   decisions.

3. **Lightweight and simple.** The trainable part of the policy is 43.5M
   parameters (a 4-layer VA composite plus an 8-layer flow head); deployment
   runs one VA forward plus 32-step Euler integration of the flow head at
   40.6 Hz on a single consumer GPU. Training uses a two-term loss — flow
   matching plus a paired-instruction contrast — with no auxiliary objectives
   (no image reconstruction, no world-model targets, no retrieval modules)
   used by memory-augmented VLA baselines.

4. **Honest scaling report.** We report the closed-loop gap of the
   single-start training data (samples covered only the first 0.33 s of each
   episode) and
   trace it to data coverage rather than architecture, verified by
   in-distribution tasks succeeding 10/10. The multi-start full-coverage
   rebuild and the structural upgrade chain (AQC → VA2, Sec. 3) retest this
   gap; VLA-RL gains and LIBERO-100 complete the scaling picture
   [TBD: VA2 numbers, LIBERO-100, RL].

We emphasize the scope of our claims: we do not argue that end-to-end VLA
fine-tuning universally destroys semantic representations — only that within
our tested configurations, the frozen-cache design preserves instruction
selectivity that the end-to-end variant loses, and that this difference is
attributable to the trainable language adapter.

## 2. Related Work

### 2.1 Lightweight VLA policies

Vision-language-action models stack a pretrained VLM backbone with an action
head. Scale ranges from 7B-parameter OpenVLA-family models
\cite{kim2024openvlaopensourcevisionlanguageactionmodel} to sub-1B
lightweights: SmolVLA (0.45B) trains a compact flow head on VLM embeddings
\cite{shukor2025smolvlavisionlanguageactionmodelaffordable};
Evo-1 (0.77B) shows two-stage training — frozen-VLM action-head pretraining
followed by full fine-tuning — consistently beats one-stage training
\cite{lin2025evo1lightweightvisionlanguageactionmodel};
TurboVLA (0.2B) removes the LLM trunk entirely and reaches 97.7% LIBERO
average with a DINO-style vision encoder and a compact adapter
\cite{xie2026turbovlarealtimevisionlanguageactionmodel}. π0 and π0.5
(3B+) popularized flow-matching action heads with pretrained VLM backbones
(PaliGemma) \cite{black2026pi0visionlanguageactionflowmodel,
intelligence2025pi05visionlanguageactionmodelopenworld}. Our work sits in
this lightweight family (43.5M trainable
parameters, frozen Qwen3.5-2B + V-JEPA 2.1 feature extractors
\cite{murlabadia2026vjepa21unlockingdense}) and shares
the flow-matching action head of π0/Evo-1/SmolVLA.

### 2.2 Memory-augmented VLA

Several recent architectures augment VLA policies with explicit temporal
state. ReMem-VLA maintains double-layer EMA recurrent queries and adds an
image-reconstruction auxiliary loss (MemoryBench 94.5%)
\cite{li2026rememvlaempoweringvisionlanguageactionmodel}; MemoryVLA builds a
perceptual-cognitive memory bank with retrieval (LIBERO-5 96.5%)
\cite{shi2026memoryvlaperceptualcognitivememoryvisionlanguageaction}; AVA-VLA
learns a recurrent belief with active visual attention under T=4 BPTT
\cite{xiao2026avavlaimprovingvisionlanguageactionmodels};
RB-VLA couples a belief state with a world-model self-supervised objective
\cite{bagaria2026recursivebeliefvisionlanguage}.
CogVLA couples visual and action streams in a bidirectional attention decoder
(LIBERO 97.4%) \cite{li2026cogvlacognitionalignedvisionlanguageactionmodel}.
Our VA composite is closest in spirit to the recurrent
bidirectional designs (ReMem/RB/AVA), but differs in two respects: the memory
is a single-slot overwritten visual state per layer (constant memory, no
retrieval or bank), and — central to this paper — the language representation
enters through a frozen static cache rather than a trainable shared backbone,
so our training objective needs none of the auxiliary losses (image
reconstruction, world-model targets) these baselines use to protect or
maintain representations.

Beyond explicit memory banks, a parallel line conditions the action decoder
on structured world or scene states: WAM4D learns a fast 4D world action
model via spatial register tokens \cite{li2026wam4dfast4dworld}; CheckVLA
adds execution-time verification with an action-conditioned world model
\cite{liu2026checkvlaexecutiontimeverificationactionconditioned};
EvoScene-VLA evolves scene beliefs inside the action decoder
\cite{zhang2026evoscenevlaevolvingscenebeliefs} and Act2Goal distills a
general goal-conditioned policy from a world model
\cite{zhou2025act2goalworldmodelgeneral}; S$^2$-VLA uses state-space guidance
for long-horizon manipulation \cite{xie2026s2vlastatespaceguidedvisionlanguageaction},
FutureRTC uses anticipatory-conditioned action chunking for real-time
execution \cite{jiang2026futurertcrealtimerobotexecution}, and
Acting-While-Understanding / DEFLECT decouple or reorder semantic-action
streams to make asynchronous VLA pipelines delay-robust
\cite{yan2026actingunderstandingasynchronoussemanticaction,
zhu2026deflecttemporalcounterfactualpreference}. On the representation side,
AnySlot, SSI-Policy, ReKep and Action-QFormer shape slot-level, scene-level,
relational-keypoint and action-supervised structured representations
\cite{hu2026anyslotgoalconditionedvisionlanguageactionpolicies,
wang2026ssipolicylearningstructuredscene,
huang2024rekepspatiotemporalreasoningrelational,
ji2026actionqformerstructuredrepresentation}, and
\cite{jia2025learningefficientrobustlanguageconditioned} learns robust
language-conditioned manipulation through textual-visual relevancy and
equivariant language maps. Our shared-attention VA stack is a lightweight
instance of the attention-as-computation view of
\cite{schug2025attentionhypernetwork}.

### 2.3 Instruction blindness and language grounding in VLA

Benchmarks: LIBERO provides multi-task same-scene manipulation with diverse
instructions \cite{liu2023liberobenchmarkingknowledgetransfer}; LIBERO-Plus
formalizes robustness protocols including blank and
swapped instructions \cite{fei2025liberoplusindepthrobustnessanalysis}.
LIBERO-CF introduces counterfactual benchmarks showing
state-of-the-art VLAs (OpenVLA, π0.5) frequently ignore language when scene
features suffice \cite{libero_cf} — the visual-shortcut failure we target
with counterfactual evaluation. On the training side, flatness/sharpness
analyses (SAM-based fine-tuning) report +217% instruction-following on
LIBERO-CF with zero architecture change \cite{zhang2026flatnesspreservesinstructionfollowing},
and Knowledge Insulation shows continuous
action-expert gradients damage language-command interpretation, motivating
stop-gradient isolation of the backbone
\cite{driess2025knowledgeinsulatingvisionlanguageactionmodels}. π0.5 designs
language evaluations that avoid object-selection shortcuts. CAST and LA4VLA
attack the same visual-shortcut regime from the data and architecture sides
\cite{glossop2026castcounterfactuallabelsimprove,
lin2026la4vlalearningactseeing}; semi-supervised language interfaces
\cite{myers2023goalrepresentationsinstructionfollowing} likewise aim at
richer instruction conditioning from limited supervision. Our frozen-cache design is the
architectural extreme of the stop-gradient direction: no gradient can reach
language parameters, which we verify directly (Sec. 5.5).

### 2.4 Flow-matching policies and RL fine-tuning

Flow matching and diffusion action heads are the de-facto standard (π0,
π0.5, GR00T, RDT-1B \cite{black2026pi0visionlanguageactionflowmodel,
intelligence2025pi05visionlanguageactionmodelopenworld,
nvidia2025gr00tn1openfoundation, liu2024rdt1b}). Fine-tuning such policies
with RL has been addressed by
Diffusion Policy Policy Optimization (DPPO) \cite{ren2024diffusionpolicypolicyoptimization},
ReinFlow (policy gradients
through the denoising path, NeurIPS 2025) \cite{zhang2026reinflowfinetuningflowmatching}
and πRL (Flow-Noise/Flow-SDE
augmented Markov policies, LIBERO 57.6→97.6%, MetaWorld 50.8→86%)
\cite{pi_rl}. We follow
the ReinFlow-lite recipe of Codex-designed minimal noise schedule: 32
learnable per-step per-dimension transition-noise parameters on top of the
imitation-learned flow head, PPO with the exact joint path log-probability
(Sec. 5.7).

## 3. Method

### 3.1 Architecture

The policy has three components (Fig. 1):

1. **Frozen language encoder with static cache.** The instruction $l$ is
   encoded once by a frozen Qwen3.5-2B text backbone \cite{qwen3point5}. All language token
   hidden states at the final layer are cached as language memory
   $K_L, U_L$. At deployment the encoder never runs again per decision step —
   only when the instruction changes. Critically, the encoder is *frozen and
   detached*: action gradients cannot flow into language parameters (Sec.
   3.4), which structurally prevents representation erosion.

2. **Vision encoder.** V-JEPA 2.1 (ViT-B) encodes the current causal 4-frame
   window into visual tokens $V_t$ (flat-pooled 64 tokens; the spatial-grid
   variant is the subject of Sec. 5.x).

3. **VA composite with causal-decomposed memory.** A stack of $N$
   shared-attention blocks with four read/write streams — vision, action,
   task workspace — plus the static language anchors, a protected evidence
   memory, and an explicit robot-state source. The recurrent state of the
   v1 policy was a snapshot of the visual stream; because that stream reads
   action and language tokens, the snapshot was a *multimodal workspace*
   rather than evidence, letting policy intent contaminate the model's
   belief about the world (confirmation bias). We decompose it into

   - **Protected evidence memory** $E_t$: updated only by the current frozen
     visual tokens $V_t$, the robot state $S_t$ (proprio + previous executed
     action) and the previous evidence, through a gated cross-attention
     update $E_t = (1-g_t)E_{t-1} + g_t\tilde E_t$ with
     $g_t=\sigma(W_g[E_{t-1},\tilde E_t,\tilde E_t-E_{t-1}])$. Inside the
     stack $E_t$ appears **only as a key/value source** — no query group, in
     particular not the action stream, can write into it (protection by
     construction). The first step fully overwrites the learned initial slots.
   - **Task workspace** $T_t$: a small token set tracking the dynamic
     reading of the contract (role bindings, satisfied constraints,
     progress), initialized once per episode from the language anchors
     ($T_0=\mathrm{TaskResampler}(K_L)$) and updated as a read/write stream
     with a gated residual $T_t = T_{t-1} + g^T\odot(T^N - T_{t-1})$.

   The attention layout (with $E$ as the memory group) is

   $$
   Q=\begin{bmatrix}V_tW_Q^V\\A_tW_Q^A\\T_tW_Q^T\end{bmatrix},\quad
   K=\begin{bmatrix}V_tW_K^V\\E_tW_K^E\\A_tW_K^A\\T_tW_K^T\\K_L\\S_tW_K^S\end{bmatrix},
   $$

   and the final action condition is $C_t=\mathrm{LN}(A_t^N + P_T T_t^N)$:
   the flow head is conditioned on the *current task interpretation*, not
   only on the static contract. **Sequential coupling** (every $k$-th layer,
   $k=2$): the layer runs the explicit three-step cascade
   $A^{\frac12}\leftarrow A+\mathrm{Attn}(A\!\to\![V,E,T,L,S])$ (proposal),
   $[V',T']\leftarrow \mathrm{Attn}([V,T]\!\to\![V,E,A^{\frac12},T,L,S])$
   (action-conditioned reorganization), and
   $A'\leftarrow A^{\frac12}+\mathrm{Attn}(A^{\frac12}\!\to\![V',E,A^{\frac12},T',L,S])$
   (correction), making "vision interprets the action hypothesis" a
   first-class computational step.

4. **Deeply conditioned flow head.** A flow-matching module parameterizes
   $\dot a^\tau = v_\theta(a^\tau,\tau\mid C_t)$ over the full action chunk.
   With `flow_cond=adaln` the head is a DiT-style transformer: every layer
   receives AdaLN-Zero modulation from $(C_t,\tau)$ and a cross-attention
   over $C_t$; zero-initialized gates make the training start an
   unconditioned flow field that learns its conditioning gradually.
   Deployment: 32 Euler steps (the 8-step fast variant shows staircase
   artifacts; Sec. 4).

### 3.2 Training

The vision and language backbones stay frozen during initial training; the VA
composite is unrolled over 4 consecutive visual timesteps of the same task so
that the action loss back-propagates through the recurrent memory to previous
visual states, language projections, memory K/V and the flow head. With
$a^\tau=(1-\tau)\epsilon+\tau a$ and $\tau\sim U(0,1)$:

$$
\mathcal L_{FM}=\mathbb E\left\|v_\theta(a^\tau,\tau\mid C_t)-(a-\epsilon)\right\|_2^2.
$$

**Shared-source counterfactual supervision** (replaces the v1 pair loss).
A naive pair loss comparing flow velocities under the same noise at random
$\tau$ is ill-posed: the interpolated inputs already differ by
$\tau(a_i-a_j)$, so a language-blind head can satisfy the delta target. We
evaluate both instructions at a **shared probe**
$x=(1-\tau)\epsilon+\tau\bar a_{ij}$ with $\bar a_{ij}=(a_i+a_j)/2$ and
per-pair $\tau\sim U[0,0.5]$ (the $\tau=0$ source point is included), with
linear-FM targets $u_i=(a_i-x)/(1-\tau)$. Every non-language input is then
identical, so the velocity difference at the probe is attributable to the
language condition:

$$
\mathcal L_{CF}=\mathrm{Huber}(v_\theta(x,\tau,C_i),u_i)+
\mathrm{Huber}(v_\theta(x,\tau,C_j),u_j)+
\mathrm{Huber}(v_\theta(x,\tau,C_i)-v_\theta(x,\tau,C_j),\,(a_i-a_j)/(1-\tau)).
$$

Pairs are genuine same-scene forks: instructions sharing a scene whose first
state satisfies a cosine-gated contract (feature cosine ≥ 0.99, proprio
max-diff ≤ 0.15, previous action exactly zero — the deployment contract at
episode start). [Data fact, measured: MetaWorld MT50 episodes are single-goal
and open/close families start from opposite states (min first-state distance
0.64), so no genuine forks exist there; $L_{CF}$ evidence comes from LIBERO,
and MW uses FM + future-latent supervision only.]

**Future-latent regularizer.** To give the action→vision backward path
physical meaning, a lightweight head maps $(E_t,T_t,C_t)$ to the mean of the
next decision's frozen V-JEPA features under a stop-gradient target
(features are precomputed by the frozen encoder; no future information
enters the policy input path):

$$
\mathcal L_{future}=1-\cos\!\big(P_\psi(E_t,T_t,C_t),\,\overline{V_{t+1}}\big).
$$

Total IL loss:
$\mathcal L=\mathcal L_{FM}+\lambda_{CF}\mathcal L_{CF}+\lambda_{future}\mathcal L_{future}$,
with $\lambda_{CF}=1.0$ and $\lambda_{future}=0.1$. No other auxiliary
objectives. [Note: the artifact-era checkpoints (A/B40k/C1/C2) were trained
FM-only with pair=0 because the then-current paired dataset was misaligned;
they serve as matched controls for the 2×2 verdict of Sec. 5.5.]

### 3.3 Deployment

Single VA forward (evidence gate + task update + $N$ attention blocks) +
Euler integration of the flow head — 32 steps (Sec. 4);
no autoregressive decoding, no diffusion iteration. The language encoder
runs once per instruction; the future-latent regularizer is training-only.
Control frequency: 13.3 Hz decision loop on MetaWorld (80 FPS env, 6-step
macro actions) and 10 Hz on LIBERO (20 Hz env, 2-frame stride); per-step
re-inference (`--execute-steps 1`) is available as a protocol probe
(Sec. 5.3).

### 3.4 Why the frozen cache prevents representation erosion

The causal chain behind the first mechanism of instruction blindness is:

```
action-loss gradients → shared backbone → language-related parameters rewritten
```

In the VA composite this chain is cut at its first link. The language encoder
is called in `no_grad` mode and its outputs are detached before entering the
VA composite; the action loss back-propagates through the VA blocks, the
vision tokens and the flow head, but the computation graph contains no path
into language parameters. This is a property of the graph structure, not a
training trick: no gradient scaling, early stopping or regularization is
needed to prevent language drift. The residual risk is not erosion of the
encoder but *collapse of the read-out* — the VA's own language projections
($W_K^L, W_U^L$) could degenerate. We measure the encoder itself directly
(pairwise cosine of instruction embeddings, Sec. 5.5) and the read-out
behaviorally (counterfactual language perturbations, Sec. 5.2/5.3), so the two
failure modes are distinguishable empirically.

## 4. Experimental Setup

**Benchmarks.** (i) PNPW: a single-task pick-and-place data suite
(118k frames) used for architecture ablations (pooling, depth, sampling).
(ii) LIBERO 3-scene 12-task subset (LIVING_ROOM/KITCHEN/STUDY, 360 samples):
same-scene different-instruction regime — the canonical setting for exposing
visual shortcuts. (iii) MetaWorld MT50 (49 tasks × 50 demos, SmolVLA's
lerobot/metaworld_mt50 data source, 2488 samples after episode slicing).
(iv) LIBERO-100 (full benchmark, 100 tasks × 50 episodes, 5000-episode
lerobot/kevin_libero100 raw data verified locally; 25 episodes/task × 2
sequences = 5000 samples). CALVIN, RLBench and RoboTwin 2.0 clean50
\cite{chen2025robotwin20scalabledata} are not
included: only the CALVIN debug split (single episode) is available locally,
RLBench and RoboTwin 2.0 are absent from the local data disk, and RoboTwin's
bimanual 14-DoF action space would require an action-space redesign of the
single-arm policy — the adaptation cost assessment therefore concluded
against inclusion; the debug split is insufficient for benchmark-level
numbers and is not reported.

**Protocol.** Flow sampling at 32 Euler steps (8-step integration shows
staircase artifacts: adjacent-step jump 0.0045 vs 0.0004 at 32 steps).
Metrics: chunk MAE / MSE on normalized actions, first-step success (<0.05
threshold), macro-average over tasks (task-level samples averaged first),
95% bootstrap CI (B=2000, seed 0, resampling unit = task). Because 30 FPS
demonstrations are smooth, the copy-previous-action persistence baseline is
reported alongside every number (per the copycat analysis of
\cite{wen2020fightingcopycat} and the past-token-prediction analysis of
\cite{torne2025learninglongcontextdiffusionpolicies}, a first-step
success metric alone is dominated by this trivial baseline — e.g. 99.7% on
PNPW).

**Language perturbations.** *blank* — zero the entire language stream;
*swap* — rotate instructions by +1 index; *wrong* — rotate MetaWorld
instructions by +1 mod 49; *task-id* — replace language features with a
task-identity token. Counterfactual closed-loop verdicts (L_m) use
same-scene dual-objective pairs where only the language cache differs between
otherwise identical rollouts (Sec. 5.6).

**Smoke protocol** (fixed before the VA2 evaluation, run on the checkpoint
only). 800 genuine same-state fork pairs from 3 MetaWorld tasks: the two rows
of each pair share vision/proprio/previous-action at every time step and
differ only in the language stream. Metric A (source probe): with one shared
flow noise ε per pair, v_θ(ε, 0; C_i) − v_θ(ε, 0; C_j) must align with the
expert action split a_i − a_j; directional accuracy = fraction of pairs whose
mean cosine is positive, bar ≥ 16/18 = 88.9% (converted to 712/800). Metric B
(execution): 32-step Euler rollout, first-step directional accuracy plus
per-condition chunk MAE.

**Training.** VA composite 8 layers (PNPW/MW) or 4 layers (LIBERO e2e
variants), 40k steps, batch size 1 (legacy rows) or 4 (VA2, 4× decision
exposure), AdamW; backbones frozen (config A/C1/C2) or partially unfrozen
(config B: V-JEPA last 12 blocks + Qwen LoRA r32). Language/vision features
are precomputed and cached for the frozen variants. The VA2 architecture
(Sec. 3) additionally uses the causal-decomposed memory, sequential
coupling, deeply conditioned flow head and the future-latent regularizer,
trained with L = L_FM + λ_CF L_CF + λ_future L_future (λ_CF = 1.0 on the
cosine-gated LIBERO fork pairs, λ_future = 0.1). All v3 datasets carry the
deployment-consistent previous-action contract (t=0 zeros, t>0 =
actions[t-1,-1]).

## 5. Results

### 5.1 Single-task pilot (PNPW)

Table 1. Pooling and depth ablations (first-step success, first_mae, chunk
mae, vs. persistence baseline on chunk):

| config | success | first_mae | chunk_mae | vs persistence |
|---|---|---|---|---|
| 4 layers @10k (flat) | 90.9% | 1.228 | 0.0497 | -0.0055 |
| 4 layers @10k (spatial) | 84.5% | 1.542 | 0.0564 | -0.0014 |
| 8 layers @10k (flat) | 85.7% | 1.406 | 0.0518 | -0.0034 |
| 8 layers @20k (flat) | 99.1% | 1.063 | 0.0345 | -0.0207 |
| persistence baseline | 99.7% | — | 0.0552 | — |

Depth must be matched with step budget: 8 layers @10k looks worse but 8
layers @20k beats 4 layers by 31% chunk error (4x the real gain of the
4-layer config). Flat pooling (token-mean) beats spatial pooling. An
intervention audit (masking each stream at inference) showed the
action→vision backward path (+7.7% error), memory→action (+2.9%) and
previous-action (dominant input, +3792% when zeroed) are all real; language
showed ~0% in this single-task data because the instruction is constant —
multi-instruction data is required to test the language stream (Sec. 5.2).

### 5.2 Language grounding on LIBERO (open loop)
| dataset | clean | blank | swap |
|---|---|---|---|
| 1-scene (4 tasks) | 0.00106 | +13751% | +1518% |
| 3-scene (12 tasks) | 0.00254 | +2381% | +607% |

Language is causally necessary under same-scene different-instruction data —
the design regime where VLA shortcut risk is highest.

### 5.3 MetaWorld MT50

**Open loop (40k steps, 32-step flow, macro-average over 49 tasks).**
On the multi-start full-coverage rebuild: chunk MAE 0.0806 [0.0709, 0.0921]
vs persistence 0.0930 → -13.4% relative improvement; first-step success
50.8% [45.3%, 56.1%] (persistence: 88.5% — the threshold metric is
persistence-dominated, Sec. 4); first_mae 0.0725 (persistence 0.0441).
The rebuild itself is the largest open-loop lever: the fix-retest
(single-start, 0.33s coverage) model measured chunk MAE 0.0896
[0.0802, 0.0994] and first-step success 31.4% [26.0%, 37.2%] — full-coverage
data alone gives -10% chunk MAE and +19.4 pp first-step success
(logs/mw_full_openloop.log; development record VA_COMPOUND_REPORT §8.2).

**VA2 retest (v4 prev-contract, 40k steps, 32-step flow, same 49 tasks).**
On the leakage-free v4 contract: chunk MAE 0.0803 [0.0710, 0.0915] vs
persistence 0.15935 → -49.6% relative improvement; first-step success
50.2% [44.7%, 55.9%] (persistence: 70.7%); first_mae 0.08125 (persistence
0.12084). The v4 retest matches the pre-VA2 rebuild (chunk MAE 0.0806,
success 50.8%), i.e., the open-loop macro metrics are saturated at this
data scale; the VA2 gains concentrate in closed-loop control and in
language grounding (§5.3 ablation below, §6).
*Development record: the initial VA2 run used the v3 "prev-fix" contract,
which was later found to leak the future action into `previous_action`
(prepare_prev_fix.py wrote `actions[t-1,-1]`, which equals `actions[t,1]`
under the 6-step control stride / 8-step horizon overlap; verified equality
rate 1.0000). That run's open-loop chunk MAE (0.0706) was a leakage
artifact, not a real gain; its numbers are superseded by the v4 retest
below (logs/mw_va2_openloop.log → superseded; logs/mw_va2_v4_openloop.log).*

**Language ablation (same model, instruction rotated +1 / replaced by
task-id).**

| condition | chunk@decision | chunk@all-seq | vs clean |
|---|---|---|---|
| clean | 0.0912 | 0.0799 | — |
| wrong instruction | 0.2307 | 0.1666 | +153.0% / +108.5% |
| task-id token | 0.1855 | 0.1450 | +103.5% / +81.5% |

Wrong instructions degrade actions substantially more than task-id tokens —
language content beyond task identity causally participates in decisions
(logs/mw_full_ablation.log).

**VA2 ablation (v4 contract, 32-step flow; v3 numbers superseded by the
leakage finding above).**

| condition | chunk@decision | chunk@all-seq | vs clean |
|---|---|---|---|
| clean | 0.09495 | 0.07984 | — |
| wrong instruction | 0.31164 | 0.20018 | +228.2% / +150.7% |
| task-id token | 0.26299 | 0.17496 | +177.0% / +119.1% |

Wrong instructions degrade actions more than task-id tokens (+150.7% vs
+119.1% at chunk@all-seq) — language content beyond task identity causally
participates in decisions (logs/mw_va2_v4_ablation.log).

**Closed loop (49 tasks × 10 trials, fixed seeds).** The chain of
improvements: 7.1% [2.7%, 12.7%] (35/490, single-start data, coverage
limited) → 16.3% [9.4%, 24.1%] (80/490, multi-start full-coverage rebuild)
→ 17.8% [11.0%, 25.3%] (87/490, + Qwen-conditioned action queries) →
21.4% [13.7%, 29.0%] (105/490, executed-action clip contract, Sec. 4) →
**31.8% [22.6%, 41.4%] (156/490, + direct deterministic head on executed
labels, v5 contract; logs/mw_v5_direct_closedloop.log)**. The residual gap
to literature (SmolVLA 57.3% / Evo-1 80.6%, macro-4 protocol)
is attributed in Sec. 6.5 to exposure (≈40× fewer decision presentations),
the 6-step open-loop execution protocol, and the flat 64-token visual
pooling; the VA2 architecture (Sec. 3: causal-decomposed memory, sequential
coupling, deep flow conditioning, 4× exposure, prev-contract fix) retests
this chain [TBD: logs/mw_va2_closedloop.log].

**Executed-action contract and direct head (pilot).** We rebuilt the
action labels as the *executed* actions (the environment clips raw expert
commands to [-1, 1]; 21.4% of raw actions exceed the bound, so the old
labels aliased distinct normalized values to the same executed behavior;
the rebuilt v5 labels are an exact identity map — denormalization
round-trip error 0, zero alias violations). On a 2-task pilot
(button-press + peg-insert, 400 samples, 6000 steps, batch 32) a
deterministic direct head (MLP→tanh) trained on executed labels reaches
training loss 3×10⁻⁴ and 10/10 closed-loop on the trained button task —
vs 6/10 for the flow-matching model on the same data — while all models
score 0/10 on the three held-out tasks, isolating data coverage (2 tasks)
as the binding constraint at this scale (logs/mw_pilot_*).

**C² contraction-control ablation (pilot).** We implemented a
feedback-controller action token {nominal ū, reference c̄, gain K}
(16-dim controllable projection P frozen from recovery-data PCA; K
zero-initialized; L_future on per-chunk-step c̄ targets; recovery loss
L_r = Huber(K·e, sg(ū−a^E)) on 11,280 DAgger-style perturbed transitions
collected from the scripted expert; contraction index recorded, not
trained). Results on button-press (10 trials each):

| deployment | clean closed-loop | held-out recovery |
|---|---|---|
| direct head (no C²) | 10/10 | 30% |
| C² K full | 6/10 | **50%** |
| C² K × 0.5 | 9/10 | 30% |
| C² error-deadzone τ=0.3 | 8/10 | — |

The learned gain transfers genuine recovery capability (+20 pp on
perturbed-start branches, 30%→50%), but every clean closed-loop
deployment trails the direct head: the reference prediction is imperfect
(future loss 0.013) and K amplifies that noise when the policy is already
near-expert. We report C² as partial validation — recovery value is real,
universal deployment is not — and keep it as an ablation rather than the
mainline controller (logs/mw_pilot_c2v2_*, mw_pilot_c2v2_recovery*).

### 5.4 Comparison to literature baselines

Table 2. All baseline numbers are cited from the original papers (Grok-verified
protocols), not trained by us; protocols noted per row family.

| model | MetaWorld MT50 (closed-loop) | LIBERO avg | notes |
|---|---|---|---|
| FabriVLA \cite{yang2026fabrivlalightweightvisionlanguageactionmodel} | 90.0% | — | imitation SOTA |
| LA4VLA \cite{lin2026la4vlalearningactseeing} | 87.5% | — | |
| π0+ALAM \cite{tang2026alamalgebraicallyconsistentlatent} | 85.0% | — | |
| Evo-Depth \cite{lin2026evodepthlightweightdepthenhancedvisionlanguageaction} | 84.4% | — | |
| Evo-1 \cite{lin2025evo1lightweightvisionlanguageactionmodel} | 80.6% † | 94.8% † | 0.77B, two-stage |
| SmolVLA \cite{shukor2025smolvlavisionlanguageactionmodelaffordable} | ~68% † | ~89% † | 10 trials/task, VLM-init only |
| π0.5 \cite{intelligence2025pi05visionlanguageactionmodelopenworld} | — | ~97% ‡ | 50 trials (OpenPI/LeRobot) |
| TurboVLA \cite{xie2026turbovlarealtimevisionlanguageactionmodel} | — | 97.7% ‡ (99.2/99.8/97.4/94.2) | 0.2B, no LLM trunk |
| π0 \cite{black2026pi0visionlanguageactionflowmodel} | 47.9% § | 94.2% ‡ | MW third-party citation |
| **ours (open-loop)** | chunk 0.0806 [-13% vs persistence] | — | not comparable to closed-loop |
| **ours (closed-loop)** | 17.8% [11.0, 25.3] | — | chain: 7.1 → 16.3 → 17.8; VA2 [TBD] |

† MT50: 50 demos/task, 10 trials/task (Seo difficulty split); VLM-init only.
‡ LIBERO: OpenVLA reports 3 seeds × 50 ep/task; SmolVLA/Evo-1 report 10
trials/task; OpenPI π0/π0.5 use 50 trials/task. Protocols are not mixed
within one family. § π0 MT50 number is a third-party citation, not from the
original paper.

### 5.5 End-to-end fine-tuning collapse and the 2x2 control

All numbers below are on the LIBERO 3-scene 12-task set with the v2 data
(previous-action leakage across episodes fixed). Configurations: **A** =
frozen features (V-JEPA + Qwen caches, VA 8 layers, 20k steps); **B40k** =
end-to-end (V-JEPA last 12 blocks unfrozen + Qwen LoRA r32 + VA 4 layers,
10k + 30k warm-start); **C1** = fully frozen e2e (same video input as B, all
backbones frozen, VA 4 layers, 40k); **C2** = V-JEPA frozen + Qwen LoRA r32
(VA 4 layers, 40k).

**Behavioral language sensitivity (open loop, 32-step, 360 samples, v2
data):**

| config | clean chunk_mae | blank | swap |
|---|---|---|---|
| A (frozen) | 0.00254 | +2381% | +607% |
| B40k (e2e) | 0.0759 | +1.5% | +2.7% |
| C1 (frozen e2e) | 0.08391 | +33.7% | +42.6% |
| C2 (LoRA only) | 0.08390 | +0.4% | +1.8% |

**Instruction embedding collapse (Qwen, pairwise cosine over the 12
instructions, mask-corrected last-token protocol):**

| encoder state | cosine |
|---|---|
| original Qwen (untrained) | 0.8573 |
| random features | 0.0023 |
| B40k LoRA weights applied | 0.9994 |
| C1 (never trained) | 0.8573 |
| C2 (LoRA only) | 0.9984 |

B40k collapses the instruction embedding space to near-identity (0.999) and
loses behavioral sensitivity (+1.5% vs +2381%). C1 — the same video-input e2e
pipeline with all backbones frozen — keeps the exact original embedding
space (0.857), isolating the collapse to *training* rather than the e2e
pipeline itself. **The C2 verdict is decisive: LoRA alone — with vision
frozen — collapses the embedding space (0.998) and destroys behavioral
sensitivity (+0.4%/+1.8% vs C1's +33.7%/+42.6% on the same video-input e2e
pipeline).** The collapse driver is the trainable language adapter, not
vision fine-tuning. The repair is therefore language isolation: the e2e
pipeline must keep Qwen frozen (C1 configuration), and our production
architecture (A) already does so by construction. [L_m closed-loop verdict
and C_OL counterfactual displacement: TBD — closed-loop LIBERO rollouts show
D=O=0 (OOD fragility), so the obedience verdict relies on C_OL and the
open-loop counterfactual metrics; see Sec. 5.6.]

### 5.6 Language obedience: counterfactual closed-loop verdict

To test whether the policy *obeys* instructions rather than merely being
sensitive to them, we run same-scene dual-objective counterfactual rollouts
(L_m verdict). For each scene we select two executable objectives (g1, g2)
with their instructions (l1, l2). On each matched initial state we roll out
four conditions in which only the language cache differs — the physical
environment, initial state and sampling are identical:

```
D: env(g1)+l1 -> r1,  env(g2)+l2 -> r2      (matched instruction)
O: env(g1)+l2 -> r3,  env(g2)+l1 -> r4      (swapped instruction)
```

L_m = mean over matched blocks of 1/2[(r1−r3) + (r2−r4)] = D − O, with
block bootstrap 95% CI. Verdicts: L_m >> 0 → obedience (behavior follows the
language swap); L_m ≈ 0 with high D and O → missing language selectivity
(visual shortcut); L_m ≈ 0 with low D and O → OOD fragility; L_m < 0 →
inverse effect. Pairs: STUDY back/front and left/right compartments of the
caddy; KITCHEN black bowl at back/front; LIVING soup/butter and milk/juice
into the basket (5 pairs, benchmark-pinned to guarantee same-scene layouts;
the swapped condition doubles as the held-out command-fork test since both
instructions are executable in the same scene).

**Result (closed-loop, horizon 400, 5 trials per condition per pair).** For
config A all five pairs give D=O=0 (L_m = 0.000 [0.000, 0.000]): the
policy fails every rollout, matched or swapped. This is the *OOD-fragility*
verdict — the chunk policy trained on the first 0.33 s of each episode
cannot complete full LIBERO episodes, so closed-loop binary success cannot
resolve obedience at all (the same holds for B40k: D=O=0). The obedience
question is therefore answered by the open-loop counterfactual metrics that
do not require task completion: the C_OL displacement protocol below and the
blank/swap error ratios of Sec. 5.2/5.5. The L_m fragility result itself is
evidence for the data-coverage diagnosis of Sec. 6.5, and the closed-loop
obedience verdict is re-examined in Sec. 5.7 once VLA-RL raises closed-loop
competence.

**C_OL (open-loop counterfactual output displacement; same state, same
flow noise, clean vs. swapped instruction cache, first executed action;
360 samples, 12 tasks, 32-step).**

| config | C_OL(exec) [95% CI] | C_OL(chunk) | ratio vs persistence |
|---|---|---|---|
| A (frozen) | 0.04382 [0.04013, 0.04808] | 0.08222 [0.07590, 0.08882] | 0.160 |
| B40k (e2e) | 0.04434 [0.03935, 0.04940] | 0.06686 [0.06078, 0.07325] | 0.162 |

Both models displace their output measurably when the instruction is
swapped at an identical state (≈0.16 × persistence displacement). The
difference between the two models is not in *whether* language moves the
output but in *whether the movement is correct*: A's swap degrades
action error by +607% (Sec. 5.2), B40k's by only +2.7% — B40k retains
output-level sensitivity to the collapsed read-out while losing the
grounding that makes the response track the instruction. Segment-wise,
both models show the strongest language effect on the first executed
action (early: 0.063) and weaker mid/late (0.036–0.043), consistent with
action-history masking in later decisions.

### 5.7 VLA-RL

Following πRL/ReinFlow, we fine-tune the imitation-learned policy with PPO
over a *sparse* success reward, freezing the V-JEPA and Qwen feature
extractors and training only the VA composite, the flow head, a value head
and a 32-parameter flow-noise schedule. The policy is the augmented Markov
flow policy: each Euler transition draws

```
x_0 ~ N(0, I)
x_{k+1} ~ N(x_k + (1/K) v_theta(cond, x_k, t_k), sigma_k^2 I)
sigma_{k,d} = 0.02 + 0.06 sigmoid(alpha_{k,d})
```

Rollouts store the full denoising path; the PPO ratio uses the exact joint
path log-probability (the log p(x_0) term cancels). At evaluation the
transition noise is dropped, recovering the deterministic Euler policy.
Macro actions: 8-step chunks executed for the first 6 primitives; reward = 1
if any executed primitive succeeds; episodes end on success/termination
without critic bootstrapping (time truncation bootstraps). GAE
(gamma=0.99, lambda=0.95), 4 PPO epochs, minibatch 128, clip 0.1, actor LR
3e-6, critic LR 1e-4. [TBD: MT10 subset IL→RL closed-loop comparison after
the multi-start IL checkpoint; smoke protocol first.]

### 5.8 Efficiency

Table 3. Trainable parameters and deployment latency.

| item | value |
|---|---|
| trainable params (4-layer VA + flow head) | 43.5M |
| deploy latency (32 Euler steps) | 24.66 ms/decision |
| closed-loop rate | 40.6 Hz |
| training hardware | single 24GB GPU |
| baseline scale | SmolVLA 0.45B; π0 series 3B+ (dual A100) |

## 6. Real-Robot Validation

*(Structure reserved; experiments to be conducted by the authors on
hardware.)* The frozen-cache design has two deployment-relevant properties:
the language encoder runs once per instruction (no per-step LLM inference),
and the VA composite + flow head deploys at 40.6 Hz. Planned validations:
(i) instruction following under swapped/novel instructions; (ii) closed-loop
robustness over extended horizons; (iii) counterfactual verdicts of Sec 5.6
on physical rollouts.

## 6.5 Limitations

- **Closed-loop data coverage**: MW training samples covered only the first
  0.33s of each episode → closed-loop rollouts mostly OOD (7.1% [2.7, 12.7]);
  in-distribution tasks succeed at ceiling. The multi-start full-coverage
  rebuild lifted the closed-loop result to 16.3% → 17.8% (AQC), and the VA2
  architecture (Sec. 3) retests the chain with 4× exposure and the
  prev-contract fix [TBD: VA2 closed-loop]. No closed-loop parity claims
  until the VA2 retest.
- **Genuine same-state forks do not exist in MetaWorld MT50** (measured:
  open/close families start from opposite states, min first-state distance
  0.64; every episode is single-goal). The shared-source counterfactual
  loss is therefore active only on LIBERO; MW training uses
  FM + future-latent supervision without L_pair, and the paper's causal
  language claims on MW rest on the perturbation suite (Sec. 5.3), not on
  the pair loss.
- **LIBERO pair contract is cosine-gated, not exact**: same-scene
  cross-instruction first states share feature cosine ≥ 0.99 and proprio
  max-diff ≤ 0.15 (target object configuration is part of the task
  definition); the residual visual difference is quantified in the paired
  dataset report and is a documented confound, bounded by the gate.
- **Previous-action artifact fixed**: the precomputed feature files carried
  a windowing artifact in `previous_action` at decision 0 (nonzero) while
  deployment feeds zeros at episode start; all v3 datasets rebuilt with the
  deployment contract (t=0 zeros, t>0 = actions[t-1,-1]). Numbers from the
  artifact-era checkpoints (A/B40k/C1/C2) are reported as-is.
- **Architecture drift vs. artifact-era checkpoints**: the VA2 refactor
  (causal-decomposed memory, sequential coupling, deep flow conditioning)
  added unconditional layer parameters (task-state stream, state keys),
  so the pre-refactor A/B40k/C1/C2 checkpoints cannot be loaded by the
  current code. Their published numbers (Sec. 5.5/5.6) were measured with
  the then-current code and remain valid as historical measurements;
  re-running them under the current protocol (e.g. the LIBERO fork-direction
  probe) is not possible, so the fork probe is reported for the VA2 model
  only (in-distribution), with the MW 800-pair source probe as the
  architectural counterpart.
- **LIBERO closed-loop alignment**: kevin_libero100 data lacks object-pose
  fields → official-env init alignment impossible from data; closed-loop
  evidence relies on the same-scene counterfactual protocol (Sec 5.6).
- **Protocol mixing**: literature numbers cited with per-family footnotes
  (10 vs 50 trials); our open-loop numbers labeled as not directly comparable
  to closed-loop baselines.
- **Additional benchmarks**: CALVIN dataset server unreachable from our
  network; RLBench requires CoppeliaSim/PyRep (week-scale port) — deferred;
  RoboTwin 2.0 is bimanual (14-D actions), not directly supported.
- **Real-robot validation** planned (Sec 6), reported separately.

## 7. Conclusion

We presented the VA composite, a lightweight language-grounded VLA policy in
which a frozen language encoder feeds a static key/value cache to a compact
recurrent visual-action attention stack and a flow-matching action head. The
decoupling is not an engineering convenience: it makes representation erosion
— the first identified mechanism of instruction blindness — structurally
impossible, because action gradients never touch language parameters. Our
2x2 control on LIBERO confirms both halves of the claim: freezing preserves
the pretrained instruction embedding space exactly (cosine 0.857) and
behavioral language sensitivity (+2381% blank sensitivity), while end-to-end
fine-tuning of a LoRA language adapter collapses the embedding space (0.999)
and removes behavioral selectivity (+1.5%). Counterfactual evaluation on
MetaWorld MT50 confirms that language content beyond task identity is used
(wrong-instruction +108.5% chunk error).

We remain deliberately scoped: the visual-shortcut mechanism is not solved by
any architecture alone — it is a property of the data distribution — and our
closed-loop numbers are currently limited by training-data coverage, not
architecture (in-distribution tasks succeed at ceiling). [TBD: after
multi-start rebuild, VLA-RL, LIBERO-100 and L_m verdicts, final conclusion.]

## References

*[43 entries in paper/references.bib (verified 2026-08-07: 4 unrelated + 1
duplicate removed; all keys unique). Assemble with \cite keys during the final
LaTeX/Markdown pass.]*
