from scripts.summarize_task35_fm_train import summarize_task35_fm_log


def test_task35_fm_summary_windows_and_save_events() -> None:
    lines = [
        "step=1 mode=bidir_va contract=single task=peg-insert-side-v3 loss=1.5 grad=1.0",
        "step=2 mode=bidir_va contract=single task=peg-insert-side-v3 loss=1.0 grad=1.0 aux_rmse=12.0px",
        "step=3 mode=bidir_va contract=single task=peg-insert-side-v3 loss=0.5 grad=1.0",
        "step=3 global_step=3 periodic checkpoint saved to ckpt.pt",
    ]
    summary = summarize_task35_fm_log(
        "\n".join(lines), total_steps=10, window=2
    )
    assert summary["latest_step"] == 3
    assert summary["latest_loss"] == 0.5
    assert summary["checkpoints_saved"] == [3]
    assert summary["windows"][0]["start"] == 1
    assert summary["windows"][0]["end"] == 2
    assert summary["windows"][0]["loss"]["mean"] == 1.25
    assert summary["windows"][0]["aux_rmse_px"]["mean"] == 12.0
    assert summary["windows"][1]["end"] == 3
    assert summary["windows"][1]["loss"]["last"] == 0.5
