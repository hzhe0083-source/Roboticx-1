# ORA0

ORA0 is a vision–action policy for MetaWorld manipulation. A bidirectional
Vision–Action (VA) stack is coupled, layer by layer, to a World Action Model
(WAM4VA). The two peers exchange delayed state; only the VA flow-matching head
emits physical actions. The main visual encoder is a frozen DINOv2 ViT-L.

The current experimental line is **hard2**: `assembly-v3` and
`door-unlock-v3`. Language tables are built from a 49-task MetaWorld
reference; the remaining 47 tasks contribute no training signal.

## Architecture

Control runs at 80 Hz. Planning stride 2 yields a 40 Hz decision rate. Each
decision predicts a full H6 action chunk (`action_dim=4`); only the first two
steps are executed and consumed by the world transition (`d → d+2`).

**Vision–Action.** An eight-layer bidirectional transformer
(`bidir_va`) conditions on language keys, visual tokens, proprioception, and a
recurrent visual memory. The action decoder is an AdaLN flow-matching head
(six residual blocks, eight Euler steps). Prefix steps 0–1 have weight 1.0;
the four tail steps have weight 0.036.

**World (WAM4VA).** Under `peer_sync_h6`, eight VA layers use seven World
stages. At stage \(i\) both peers read the committed snapshot from stage
\(i-1\) (one-stage delay). The world predictor publishes `world_message`
tokens that the next VA layer consumes as attention K/V; the last VA layer
therefore consumes the fully supervised terminal map.

**Coupling contract.** The predicted map value is live in the policy forward,
but the action loss stops at the map publication boundary. It trains the
map-to-policy projection and VA readers, while only the World objective trains
the visual predictor. This prevents Flow from using a DINO map as a latent
scratchpad.
Each optimizer step uses one VA batch and one World batch from disjoint
episodes, runs VA backward then World backward, and takes a single AdamW
update.

## Recurrent world state

WAM maintains two recurrent tensors with distinct lifetimes.

| Stream | Tensor | Persistence |
| --- | --- | --- |
| Perceptual | `world_map` | Every stage predicts the same next-decision endpoint from the current DINO last frame. The prior stage map is detached refinement context and the published candidate for the next VA layer, never a second physical-transition base. |
| Cognitive | `belief` | Persists across decisions. Writes are a per-channel gated convex combination followed by RMSNorm. Stage embeddings are added at read time and are not stored in the bank. |

Auxiliary stage losses decay as \(0.25^{7-i}\) with a floor of 0.1, so early
maps published into VA layers 0–4 remain supervised. Closed-loop evaluation
resets world state every four decisions (`--world-reset-every 4`) to match the
training unroll of `sequence_length=4`.

## Data

Windows follow the `peer_sync_h6_p2_world_windows_v1` contract: T4 / H6 / A4.
VA and World episodes are disjoint. Frame pointers are addressed by
`(environment name, in-file episode index)`. Expansion shards must be merged
into one long-trajectory file per task (`scripts/merge_longtraj_expansion.py`)
before phase-1 feature construction; passing extra shard files as additional
`--input` arguments collides with base-set indices.

| Family | Contents |
| --- | --- |
| `DATA_TAG=v1` | Original hard2 split (approximately 15 VA episodes per task) |
| `DATA_TAG=v2` | Expanded split (270 VA episodes, 216 World episodes) |
| `FRAMES_DIR` | One `metaworld_longtraj_<env>.pt` per task; v2 uses `data/frames_v2` |

## Layout

`va_compound/` is partitioned by domain. Top-level shims preserve
`from va_compound.X import …`.

```
va_compound/
├── policy/     VACompoundPolicy, end-to-end assembly
├── world/      WAM4VA, supervision, peer contracts
├── vision/     DINOv2 / V-JEPA backbones, metric ROI, frame cache
├── control/    residual servo, local control slots
├── utils/      exact resume, flow matching, statistics
└── data_parallel.py   post-backward gradient allreduce (do not wrap DDP)
```

## Setup

Do not hard-code a local virtualenv. Override interpreter and weight paths:

```bash
export PY=/opt/conda/bin/python
export VERIFY_PY=$PY
export DINO=/path/to/dinov2_vitl14_reg4.safetensors
```

Python dependencies are listed in `requirements.txt` (`torch>=2.4`). Closed-loop
evaluation additionally requires MetaWorld, MuJoCo, and OSMesa when `/dev/dri`
is unavailable. `libOSMesa` must match host glibc (an Ubuntu 22.04 container
cannot load a 24.04-built shared object).

## Training

```bash
bash scripts/run_mw_hard2_wam4va_visualmotion_peer_sync_h6_v1.sh {prepare|preflight|joint} [steps] [batch-size]
```

| Mode | Role |
| --- | --- |
| `prepare` | Build the immutable split, then run contract checks |
| `preflight` | Validate data identities, peer topology, and resume contracts |
| `joint` | Preflight, then train |

`num_workers` is 0: decoded frames reside in host memory, and forking would
duplicate that cache. On an NVIDIA L20 (45 GiB), the stable per-GPU batch is
24; 36 out-of-memorys.

**Single GPU**

```bash
DATA_TAG=v2 FRAMES_DIR=data/frames_v2 DECODE_CACHE_TASKS=2 \
  bash scripts/run_mw_hard2_wam4va_visualmotion_peer_sync_h6_v1.sh joint 30000 24
```

**Two GPUs (global batch 48 = 24 per card)**

Do not wrap the model in `DistributedDataParallel`. A peer step performs two
sequential backwards; DDP would allreduce after the first, leaving VA-only
gradients rank-local. `va_compound/data_parallel.py` averages optimizer-visible
gradients once, after both backwards. `--batch-size` is the global batch and
must be divisible by the number of GPUs.

```bash
# 25 epochs ≈ 5450 steps (10471 VA windows / 48)
DATA_TAG=v2 FRAMES_DIR=data/frames_v2 DECODE_CACHE_TASKS=1 NGPUS=2 \
  SAVE_EVERY=436 CHECKPOINT_DIR=/path/to/local/ckpts \
  bash scripts/run_mw_hard2_wam4va_visualmotion_peer_sync_h6_v1.sh joint 5450 48
```

## Evaluation

Closed loop, 10 trials per task, 40 Hz execution (`planning_stride` =
`execution_horizon` = `wmrm_cycle_steps` = 2). Checkpoints that are not
joint dual-stream P2/H6 are rejected.

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  bash scripts/eval_mw_hard2_wam4va.sh <checkpoint.pt> data/hard2_peer_h6_p2_eval_v2.pt
```

`eval_metaworld.py` defaults `--world-reset-every` to 4. That alignment is a
distribution match to the training unroll, not evidence that unbounded
world memory has been learned.

## Tests

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  "$PY" -m pytest tests/ -q --ignore=tests/test_recovery_param.py
```
