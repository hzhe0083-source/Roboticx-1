# Checkpoint protection and cleanup policy — 2026-08-20

## Never delete or move in this cleanup

- Every file under `checkpoints/protected_20260814/`.
- `checkpoints/mw_hard2_wam4va_visualmotion_peer_sync_h6_v1.smoke10.step10.pt`.
- Cap-0.2 exact-resume source, rolling checkpoint, and every immutable milestone from step 12500 through 15500.
- Task35 step 6000 exact-resume provenance checkpoint; step 12000, 15000, 18000, and 20000 acceptance checkpoints; all milestone SHA sidecars; especially the selected step15000 winner.
- H48 visual-motion lineage sources: final-v14 step1000, cycle-v16 memfix step300, joint-v1 step12000, stable-detach step12010, and staticweight2 step12074.
- Task35 ROI checkpoint, metric-v6 archive target and alias, `mw_va2_v4_40k.pt`, and report-backed Stage-B/final checkpoints.
- All files under `logs/`.
- Current H6 peer source/train/eval/split artifacts and all contract-bound manifests.

The exact protected SHA snapshot is stored in `sha256/protected_before.sha256` and verified after cleanup.

## Archived, not deleted

- Failed or negative-result checkpoints that retain scientific comparison value.
- Current experiment diagnostics, including PASS and NO-GO reports.
- Historical names remain available through relative symlinks when tracked scripts still use the old checkpoint path.

## Deferred pending a second owner review

- Task35 Direct 100-step variants.
- LIBERO intermediate checkpoints.
- PNPW VA8/multitask predecessors.
- DINO-action fine-tune branches beyond the file explicitly archived in this cleanup.
- Any byte-identical checkpoint whose distinct historical path is still referenced.

## Logs

No file under `logs/` may be deleted, moved, renamed, truncated, or rewritten by the cleanup. The pre-cleanup inventory is stored in `sha256/logs_before.tsv` and records paths and sizes. Post-cleanup verification found the same 549 paths with no missing or extra files. Two monitor outputs grew by normal system-cron append activity at 2026-08-20 15:20:03 +0800: `task35_fm_train_monitor.cron.log` (+2,099 bytes) and `task35_fm_train_monitor.history.jsonl` (+147 bytes). The other 547 paths retained their recorded sizes. The snapshot does not claim content hashes for logs.
