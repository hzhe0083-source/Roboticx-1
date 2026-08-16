# GOAL: two-task WAM4VA 10k world + 20k joint

## Tasks
- 0 assembly-v3
- 16 door-unlock-v3

## Recipe
1. Filter H48 windows to local JPEG episode lengths (`door-unlock-v3` uses `_fixed`; overflowing refs dropped).
2. Stage 1 `--wmrm-only` 10k, handshake off, last-frame DINO `16x16x1024`, batch 12. Keep only `checkpoints/mw_hard2_wam4va_world_10k.pt`.
3. Stage 2 resume, handshake on, VA+FM+WAM 20k, batch 6. Save every 1k as `checkpoints/mw_hard2_wam4va_joint_s{k}.pt` plus latest `..._joint.pt`.

## Launch
```bash
bash scripts/run_mw_hard2_wam4va_10k20k.sh
```

## Do not
- Use mean-pooled DINO as the world target.
- Train a 1024→16 compressor.
- Enable `--wmrm-only` on stage 2.
- Train door-unlock windows whose frame index exceeds the local `_fixed` file.
