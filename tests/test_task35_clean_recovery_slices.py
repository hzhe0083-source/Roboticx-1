from __future__ import annotations

import numpy as np
import pytest

from scripts.diag_task35_clean_recovery_slices import (
    assemble_last_decision_dense,
    slice_geometry,
)


def test_assemble_last_decision_dense_keeps_batch_and_frame_order() -> None:
    block11 = np.zeros((6, 256, 1024), dtype=np.float16)
    block23 = np.zeros((6, 256, 1024), dtype=np.float16)
    block11[1, 0, 0] = 1.0
    block11[4, 0, 1] = 2.0
    block23[1, 0, 2] = 3.0
    block23[4, 0, 3] = 4.0
    rows = np.array([[1, 4]], dtype=np.int64)
    dense5, dense11 = assemble_last_decision_dense(block11, block23, rows)
    assert tuple(dense5.shape) == (1, 512, 1024)
    assert float(dense5[0, 0, 0]) == 1.0
    assert float(dense5[0, 256, 1]) == 2.0
    assert float(dense11[0, 0, 2]) == 3.0
    assert float(dense11[0, 256, 3]) == 4.0


def test_slice_geometry_separates_clean_and_recovery_pair_distance() -> None:
    p = np.zeros((3, 4, 2), dtype=np.float32)
    p[0, 3] = (0.10, 0.00)
    p[0, 2] = (0.00, 0.00)
    p[1, 3] = (0.30, 0.00)
    p[1, 2] = (0.00, 0.00)
    vis = np.ones((3, 4), dtype=np.float32)
    vis[2] = 0.0
    report = slice_geometry(p, vis, np.array([0, 1, 0], dtype=np.uint8))
    assert report["clean"]["n"] == 2
    assert report["recovery"]["n"] == 1
    assert report["clean"]["pegHead_hole_px"]["n"] == 1
    assert report["clean"]["pegHead_hole_px"]["mean"] == pytest.approx(48.0)
    assert report["recovery"]["pegHead_hole_px"]["mean"] == pytest.approx(144.0)
    assert report["clean"]["role_std_px"]["pegHead"]["n"] == 1
    assert report["clean"]["role_std_px"]["pegHead"]["std_x_px"] == pytest.approx(0.0)
