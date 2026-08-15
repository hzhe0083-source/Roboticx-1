from pathlib import Path

from scripts.list_task35_fm_candidates import infer_step


def test_infer_step_from_archived_filename() -> None:
    assert infer_step(Path("task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1_step6000.pt")) == 6000
    assert infer_step(Path("task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1.pt")) is None
