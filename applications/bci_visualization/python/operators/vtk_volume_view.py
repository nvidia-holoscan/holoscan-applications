"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import threading
from typing import ClassVar

import cupy as cp
import holoscan as hs
import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

WINDOW_TITLE = "Kernel Flow BCI Real-Time Reconstruction and Visualization"


class VTKVolumeViewOp(hs.Operator):
    """Render the latest volume in an interactive VTK window, driven by its own clock.

    The render clock keeps mouse interaction responsive while no new volume arrives.
    """

    volume = hs.Input("volume", hs.pyobject)
    render_hz = hs.Param(30, description="Window refresh rate.", range=(1, 240))
    width = hs.Param(1024, description="Window width in pixels.")
    height = hs.Param(768, description="Window height in pixels.")
    headless = hs.Param(False, description="Render offscreen and stop when the replay ends.")
    simulate_drag = hs.Param(False, description="Inject a mouse drag after the first volume.")
    screenshot = hs.Param("", description="Write the last frame to this PNG file.")
    stop_at_last_frame = hs.Param(False, description="Stop after the requested frame limit.")
    last_frame = hs.Param(-1, description="Frame id of the final replayed sample.")
    density_min = hs.Param(-100.0, description="Lowest displayed density.")
    density_max = hs.Param(100.0, description="Highest displayed density.")

    report: ClassVar[dict] = {}

    def setup(self, spec):
        spec.input(self.volume)

    def contract(self):
        return (
            hs.Contract()
            .trigger(hs.OnClock(period=hs.time.ms(max(1, 1000 // self.render_hz))))
            .input(self.volume, hs.DrainAll())
        )

    def start(self):
        type(self).report = {
            "threads": {"start": threading.get_ident()},
            "renders": 0,
            "volumes": 0,
            "missing": {},
            "camera_moved": None,
        }
        self.image = None
        self.host = None

        self.mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.actor = vtk.vtkVolume()
        self.actor.SetMapper(self.mapper)
        self.actor.SetProperty(self._volume_property())
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.0, 0.0, 0.0)

        self.window = vtk.vtkRenderWindow()
        self.window.SetOffScreenRendering(self.headless)
        self.window.AddRenderer(self.renderer)
        self.window.SetSize(self.width, self.height)
        self.window.SetWindowName(WINDOW_TITLE)
        # The X11 interactor crashes on a headless (EGL or OSMesa) window.
        self.interactor = (
            vtk.vtkGenericRenderWindowInteractor()
            if self.headless
            else vtk.vtkRenderWindowInteractor()
        )
        self.interactor.SetRenderWindow(self.window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.closed = False
        self.interactor.AddObserver("ExitEvent", self._on_close)
        self.window.AddObserver("DeleteEvent", self._on_close)
        self.interactor.Initialize()
        self.window.Render()

    def _volume_property(self):
        # The 4.x look: blue decreases, white baseline tissue, red increases.
        low, high = self.density_min, self.density_max
        lower, upper = low + 0.4 * (high - low), low + 0.6 * (high - low)
        color = vtk.vtkColorTransferFunction()
        color.AddRGBPoint(low, 0.0, 0.0, 1.0)
        color.AddRGBPoint(lower, 0.0, 0.0, 0.5)
        color.AddRGBPoint(lower + 0.5, 1.0, 1.0, 1.0)
        color.AddRGBPoint(upper - 0.5, 1.0, 1.0, 1.0)
        color.AddRGBPoint(upper, 0.5, 0.0, 0.0)
        color.AddRGBPoint(high, 1.0, 0.0, 0.0)
        opacity = vtk.vtkPiecewiseFunction()
        opacity.AddPoint(low, 1.0)
        opacity.AddPoint(lower, 1.0)
        opacity.AddPoint(lower + 0.5, 0.5)
        opacity.AddPoint(upper - 0.5, 0.5)
        opacity.AddPoint(upper, 1.0)
        opacity.AddPoint(high, 1.0)
        prop = vtk.vtkVolumeProperty()
        prop.SetColor(color)
        prop.SetScalarOpacity(opacity)
        prop.SetInterpolationTypeToLinear()
        prop.ShadeOn()
        prop.SetAmbient(0.3)
        prop.SetDiffuse(0.7)
        prop.SetSpecular(0.2)
        return prop

    def _configure(self, volume):
        """Allocate the VTK images for the first volume and frame the camera on it."""
        shape = volume.density.shape  # (K, J, I)
        self.image = vtk.vtkImageData()
        self.image.SetDimensions(shape[2], shape[1], shape[0])
        self.image.AllocateScalars(vtk.VTK_FLOAT, 1)
        self.host = vtk_to_numpy(self.image.GetPointData().GetScalars()).reshape(shape)
        mask = vtk.vtkImageData()
        mask.SetDimensions(shape[2], shape[1], shape[0])
        mask.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        vtk_to_numpy(mask.GetPointData().GetScalars()).reshape(shape)[:] = volume.mask
        self.mapper.SetInputData(self.image)
        self.mapper.SetMaskInput(mask)
        self.mapper.SetMaskTypeToBinary()
        matrix = vtk.vtkMatrix4x4()
        for row in range(4):
            for column in range(4):
                matrix.SetElement(row, column, float(volume.affine[row, column]))
        self.actor.SetUserMatrix(matrix)
        self.renderer.AddVolume(self.actor)
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(0.0, 0.0, 0.0)
        camera.SetPosition(-1.0, 1.0, 0.6)  # left, anterior, superior in RAS anatomy space
        camera.SetViewUp(0.0, 0.0, 1.0)
        self.renderer.ResetCamera()

    def _on_close(self, *_):
        self.closed = True

    def compute(self, ctx):
        report = type(self).report
        report["threads"].setdefault("compute", threading.get_ident())
        selection = self.volume.selection()
        replay_done = False
        samples = self.volume.selected_batch()
        if samples:
            sample = samples[-1]
            volume = sample.data
            replay_done = 0 <= self.last_frame <= sample.metadata.frame_id
            if self.image is None:
                self._configure(volume)
            # The stock VTK mapper reads host memory, so copy the volume off the GPU.
            self.host[:] = cp.asnumpy(volume.density)
            self.image.Modified()
            report["volumes"] += 1
        else:
            key = "EMPTY_BATCH"
            report["missing"][key] = report["missing"].get(key, 0) + 1
        self.interactor.ProcessEvents()
        self.window.Render()
        report["renders"] += 1
        if self.simulate_drag and report["volumes"] and report["camera_moved"] is None:
            report["camera_moved"] = self._simulate_drag()
        replay_done = replay_done or selection.reason is hs.SelectionReason.END_OF_STREAM
        if self.closed or ((self.headless or self.stop_at_last_frame) and replay_done):
            ctx.request_stop()

    def _simulate_drag(self):
        camera = self.renderer.GetActiveCamera()
        before = np.array(camera.GetPosition())
        x, y = self.width // 2, self.height // 2
        self.interactor.SetEventInformation(x, y)
        self.interactor.InvokeEvent("LeftButtonPressEvent")
        for step in range(1, 11):
            self.interactor.SetEventInformation(x + 8 * step, y + 3 * step)
            self.interactor.InvokeEvent("MouseMoveEvent")
        self.interactor.InvokeEvent("LeftButtonReleaseEvent")
        self.window.Render()
        return bool(np.linalg.norm(np.array(camera.GetPosition()) - before) > 1e-3)

    def stop(self):
        type(self).report["threads"]["stop"] = threading.get_ident()
        if self.screenshot:
            capture = vtk.vtkWindowToImageFilter()
            capture.SetInput(self.window)
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(self.screenshot)
            writer.SetInputConnection(capture.GetOutputPort())
            writer.Write()
        # Observers bound to self form a cycle through VTK that Python's GC can't break.
        self.interactor.RemoveAllObservers()
        self.window.RemoveAllObservers()
        self.window.Finalize()
        self.interactor.TerminateApp()
