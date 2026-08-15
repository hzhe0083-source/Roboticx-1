import pickle
import zipfile
from pathlib import Path

from scripts.peek_task35_checkpoint_step import peek_task35_checkpoint_step


def _write_zip_checkpoint(path: Path, step: int) -> None:
    payload = pickle.dumps({"config": {}, "global_step": int(step)}, protocol=2)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ckpt.pt/data.pkl", payload)


def test_peek_reads_binint2_and_binint(tmp_path: Path) -> None:
    small = tmp_path / "step3000.pt"
    large = tmp_path / "step15000.pt"
    _write_zip_checkpoint(small, 3000)
    _write_zip_checkpoint(large, 15000)
    assert peek_task35_checkpoint_step(small) == 3000
    assert peek_task35_checkpoint_step(large) == 15000


def test_peek_real_archived_3000() -> None:
    path = Path(
        "checkpoints/task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1_step3000.pt"
    )
    if not path.is_file():
        return
    assert peek_task35_checkpoint_step(path) == 3000
