# Diagram Brief

## User Goal
- Output: one editable draw.io architecture diagram plus a rendered PNG preview.
- Audience: researchers and engineers reading the ORA0 / VA Compound prototype.
- Must communicate: the real runtime path, one `VACouplingLayer`, temporal visual memory, language caching, action output, and the paired training loop.
- Must not do: invent services, deployment infrastructure, sensors, or trainable backbones that are not present in the repository.

## Source Inventory
| id | source | type | role | priority | notes |
|---|---|---|---|---|---|
| S1 | `README.md` | documentation | content + structure | must | runtime cache/memory lifecycle, tensor contracts, frozen-backbone statement |
| S2 | `VA_COMPOUND_REPORT.md` | design note | semantics | must | bidirectional VA equations and comparison mode |
| S3 | `va_compound/backbones.py` | code | content + structure | must | Qwen text wrapper, V-JEPA input, pooling and freezing |
| S4 | `va_compound/model.py` | code | content + structure | must | exact dimensions, projections, caches, attention role mask, memory and action head |
| S5 | `train.py` | code | content + structure | must | precomputed-feature dataset, paired sampler, temporal rollout, losses and optimizer |
| S6 | `tests/` | tests | verification | should | confirms cache equivalence, memory effects, equal parameter counts and pair contract |

## Requirement Traceability
| id | requirement | source evidence | priority | planned visual encoding |
|---|---|---|---|---|
| R1 | show frozen Qwen3.5-2B language path | S1, S3 | must | blue frozen-backbone box and amber per-layer language cache |
| R2 | show frozen V-JEPA 2.1 visual path | S1, S3 | must | video tensor → encoder → bounded 64-token projection path |
| R3 | show robot state/action-token initialization | S4 | must | proprio + previous action → state projection + learned queries |
| R4 | show four shared-attention coupling layers | S2, S4 | must | trainable violet policy box plus detailed layer inset |
| R5 | show previous-layer visual memory recurrence | S1, S2, S4 | must | amber memory output with dashed temporal feedback |
| R6 | distinguish `bidir_va` and `uni_a` | S2, S4, S6 | must | explicit role-mask note in the layer inset |
| R7 | show action chunk dimensions | S1, S4 | must | green `[B,8,7]` output and controller loop |
| R8 | show actual training entry and paired losses | S1, S5 | must | separate bottom training band with precomputed feature dataset |

## Semantic Model
| id | entity or relationship | direction / cardinality | visual encoding | uncertainty |
|---|---|---|---|---|
| M1 | instruction → Qwen → hidden states | left-to-right, once per command | solid blue data arrows | none |
| M2 | hidden states → four layer caches | fan-out by policy layer | amber cache box | none |
| M3 | video window → V-JEPA → visual projection | left-to-right per observation | solid blue data arrows | none |
| M4 | state + learned queries → action tokens | left-to-right per step | solid violet arrows | none |
| M5 | V/A queries attend to V/M/A/L keys-values | fan-in inside each layer | detailed central attention path | none |
| M6 | each layer visual output → next-step same-layer memory | one-step recurrence | dashed amber feedback | none |
| M7 | final action tokens → action head → robot | left-to-right then control | green output path | physical low-level controller is outside this repo and labeled accordingly |
| M8 | paired sequences → rollout → BC + pair loss → AdamW | training pipeline | bottom red/gray band | none |

## Style Contract
| id | font | palette | stroke | icon style | density | source |
|---|---|---|---|---|---|---|
| ST1 | Helvetica, 11–24 pt hierarchy | slate text; blue frozen; amber cache/memory; violet trainable; green output; red loss | 1.5–2 px, rounded boxes; dashed temporal edges | no decorative icons | compact engineering schematic | prompt-only, repository architecture |

## Open Assumptions
| assumption | risk | how to verify |
|---|---|---|
| The requested “actual architecture” means the current ORA0 repository. | low | file names and exact tensor shapes are included in the diagram |
| “Robot controller” is an external consumer, not implemented here. | medium | box is explicitly marked “repo 外部” |
| Runtime raw video and text encoders are shown even though `train.py --data` consumes precomputed features. | low | training band explicitly states this distinction |
