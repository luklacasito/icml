"""Saved paper arrays and theory values survive the portable archive format."""

import math

import numpy as np
import pytest

from utils.results import load_npz_result, save_npz_result


def test_archive_preserves_curves_and_infinite_theory_values(tmp_path):
    path = tmp_path / "results.npz"
    curves = np.array([[1.0, 0.8], [1.1, 0.9]], dtype=np.float32)
    original = {
        "results": {"constant": {"test_loss": curves}},
        "theory": {"none": {"xi": math.inf}},
        "config": {"shape": (6, 256), "seed": np.int64(0)},
    }
    save_npz_result(path, original)
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype != object for key in archive.files)
    restored = load_npz_result(path)
    np.testing.assert_array_equal(restored["results"]["constant"]["test_loss"], curves)
    assert restored["results"]["constant"]["test_loss"].dtype == curves.dtype
    assert restored["theory"]["none"]["xi"] == math.inf
    assert restored["config"] == {"shape": (6, 256), "seed": 0}


def test_object_arrays_are_rejected_before_writing_an_unreadable_archive(tmp_path):
    path = tmp_path / "results.npz"
    with pytest.raises(ValueError, match="Object arrays"):
        save_npz_result(path, {"invalid": np.array([{"seed": 0}], dtype=object)})
    assert not path.exists()
