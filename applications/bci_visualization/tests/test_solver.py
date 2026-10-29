# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CuPy regularized solver matches a NumPy reference (needs a GPU)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import cupy as cp
from utils.reconstruction import reg_inv


class SolverTest(unittest.TestCase):
    def test_matches_numpy_dual_form_solution(self) -> None:
        rng = np.random.default_rng(0)
        jacobian = rng.normal(size=(12, 40)).astype(np.float32)
        rhs = rng.normal(size=12).astype(np.float32)
        reg = 0.1
        reg_inv._HESSIAN_CACHE.clear()
        solution = cp.asnumpy(
            reg_inv.solve_regularized_system(cp.asarray(jacobian), cp.asarray(rhs), 0, reg)
        )
        hessian = jacobian @ jacobian.T
        hessian += reg * np.sqrt(np.linalg.norm(hessian)) * np.eye(12, dtype=np.float32)
        expected = jacobian.T @ np.linalg.solve(hessian, rhs)
        np.testing.assert_allclose(solution, expected, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
