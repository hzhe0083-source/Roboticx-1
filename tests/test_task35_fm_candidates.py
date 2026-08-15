from pathlib import Path

from scripts.list_task35_fm_candidates import (
    infer_step,
    report_matches_archive,
    slice_matches_archive,
)


def test_infer_step_from_archived_filename() -> None:
    assert infer_step(Path("task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1_step6000.pt")) == 6000
    assert infer_step(Path("task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1.pt")) is None


def test_candidate_report_requires_sha_step_and_loaded_modules() -> None:
    item = {
        "ok": True,
        "loaded_modules": True,
        "sha256": "abc",
        "global_step": 6000,
        "path": "checkpoints/x_step6000.pt",
    }
    assert report_matches_archive(item, sha="abc", step=6000)
    assert not report_matches_archive(item, sha="def", step=6000)
    assert not report_matches_archive({**item, "loaded_modules": False}, sha="abc", step=6000)
    assert not report_matches_archive({**item, "global_step": 9000}, sha="abc", step=6000)
    stale_name_only = {
        "ok": True,
        "loaded_modules": True,
        "sha256": "old",
        "global_step": 6000,
        "path": "checkpoints/x_step6000.pt",
    }
    assert not report_matches_archive(stale_name_only, sha="abc", step=6000)


def test_candidate_slices_require_sha_and_step() -> None:
    payload = {
        "contract": "task35_clean_recovery_slice_v1",
        "sha256": "abc",
        "global_step": 6000,
        "slices": {},
    }
    assert slice_matches_archive(payload, sha="abc", step=6000)
    assert not slice_matches_archive(payload, sha="def", step=6000)
    assert not slice_matches_archive({**payload, "global_step": 3000}, sha="abc", step=6000)
