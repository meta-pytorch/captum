#!/usr/bin/env python3

# pyre-unsafe

import os
import tempfile

import numpy as np
import torch
from captum.concept._core.cav import CAV
from captum.concept._core.concept import Concept
from captum.testing.helpers import BaseTest


class Test(BaseTest):
    def test_cav_save_is_compatible_with_weights_only_load(self) -> None:
        concepts = [Concept(0, "striped", None), Concept(1, "random", None)]
        stats = {
            "weights": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "classes": np.array([0, 1], dtype=np.int64),
            "accs": {"0": np.float64(0.75)},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            CAV.create_cav_dir_if_missing(tmpdir, "model")
            CAV(concepts, "layer", stats, tmpdir, "model").save()
            cav_path = CAV.assemble_save_path(tmpdir, "model", concepts, "layer")

            saved = torch.load(cav_path, weights_only=True)
            self.assertEqual(saved["stats"]["weights"], [[1.0, 2.0], [3.0, 4.0]])
            self.assertEqual(saved["stats"]["classes"], [0, 1])
            self.assertEqual(saved["stats"]["accs"], {"0": 0.75})

            loaded = CAV.load(tmpdir, "model", concepts, "layer")
            assert loaded is not None
            self.assertEqual(loaded.stats, saved["stats"])

    def test_cav_load_missing_file_returns_none(self) -> None:
        concepts = [Concept(0, "striped", None)]

        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(CAV.load(tmpdir, "model", concepts, "missing"))
            self.assertFalse(
                os.path.exists(
                    CAV.assemble_save_path(tmpdir, "model", concepts, "missing")
                )
            )
