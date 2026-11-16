"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import cupy as cp
import holoscan as hs
from utils.reconstruction.reg_inv import solve_regularized_system

from .types import NormalizedSolveBatch, SolverResult

REG_DEFAULT = 0.1


class RegularizedSolverOperator(hs.Operator):
    """Run a regularized inverse solver per wavelength system."""

    batch = hs.Input("batch", hs.pyobject)
    result = hs.Output("result", hs.pyobject)
    reg = hs.Param(REG_DEFAULT, description="Regularization parameter lambda.")

    def setup(self, spec) -> None:
        spec.input(self.batch)
        spec.output(self.result)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.batch))

    def compute(self, ctx) -> None:
        sample = self.batch.receive()
        self.result.forward(sample, self._solve_batch(sample.data))

    def _solve_batch(self, batch: NormalizedSolveBatch) -> SolverResult:
        num_wavelengths = len(batch.systems)
        num_significant_voxels = int(batch.idxs_significant_voxels.size)
        num_cols_expected = batch.num_absorbers * num_significant_voxels

        # GPU-only: always use CuPy.
        result = cp.zeros(
            (num_cols_expected, num_wavelengths),
            dtype=batch.systems[0].jacobian.dtype,
        )
        for wavelength_idx, system in enumerate(batch.systems):
            if system.rhs.ndim != 1:
                raise ValueError(f"Expected 1D RHS vector, got shape {system.rhs.shape}")
            if system.jacobian.shape[1] != num_cols_expected:
                raise ValueError(
                    f"Jacobian column mismatch: expected {num_cols_expected}, "
                    f"got {system.jacobian.shape[1]} for wavelength {wavelength_idx}"
                )

            solution = solve_regularized_system(
                system.jacobian,
                system.rhs,
                wavelength_idx,
                reg=self.reg,
            )

            solution *= system.background

            if solution.shape != (num_cols_expected,):
                raise ValueError(
                    f"Solution must be 1D array with {num_cols_expected} elements, got shape {solution.shape}"
                )
            result[:, wavelength_idx] = solution

        # Reshape result to separate absorbers into mua/musp
        reshaped = result.reshape(
            (-1, batch.num_absorbers, num_wavelengths),
            order="F",
        )
        data_mua = reshaped[:, 0, :]
        data_musp = reshaped[:, 1, :]

        return SolverResult(
            data_mua=data_mua,
            data_musp=data_musp,
            idxs_significant_voxels=batch.idxs_significant_voxels,
            num_full_voxels=batch.num_full_voxels,
            wavelengths=batch.wavelengths,
            voxel_metadata=batch.voxel_metadata,
        )
