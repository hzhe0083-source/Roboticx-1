# Step 4 — Benchmark audit

| Benchmark | Use | Limitation |
|---|---|---|
| LIBERO-Long | Cheap diagnostic and ablation loop | Only 10 tasks; headline success is near saturation |
| [RoboCasa365](https://arxiv.org/abs/2603.04356) | Primary long-horizon benchmark; broad kitchens/tasks and seen/unseen composites | Higher simulation and data engineering cost |
| [VLABench](https://arxiv.org/abs/2412.18194) | Secondary semantic/common-sense stress test | Training/evaluation ecosystem is less mature |
| [BEHAVIOR-1K](https://behavior.stanford.edu/challenge/archive/2025/leaderboard.html) | Strongest compositional and recovery stress test | Full benchmark is too expensive for the first iteration; use a preregistered subset |
| [EmbodiedBench](https://arxiv.org/abs/2502.09560) | Useful for high-level MLLM agents | Mismatched to low-level VLA control as the main benchmark |

Recommended sequence: LIBERO-Long for debugging, RoboCasa365 Composite-Seen/Unseen for the paper, then an 8–10-task BEHAVIOR subset or 3–5 real-robot tasks if resources permit.
