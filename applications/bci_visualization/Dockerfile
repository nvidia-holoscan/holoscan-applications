# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This application needs only the Holoscan SDK Python wheels, so it starts from a CUDA runtime
# image instead of the SDK development image that holoscan-cli passes as BASE_IMAGE.
ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
# The graphics capability exposes the driver's OpenGL and EGL libraries to VTK.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

# CuPy JIT-compiles kernels with NVRTC and needs CUDA headers absent from the runtime image.
# Install cudart-dev matching CUDA 13.0.0 (about 24 MB installed with its dependencies).
# The devel image also adds compilers, profilers, and static math libraries we do not need;
# CuPy's [ctk] extra would install another copy of the runtime image's CUDA libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      cuda-cudart-dev-13-0=13.0.96-1 \
      python3 python3-venv git \
      libgl1 libegl1 libx11-6 libxcursor1 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
RUN python3 -m venv ${VIRTUAL_ENV}

COPY requirements.txt /tmp/repository-requirements.txt
COPY applications/bci_visualization/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir \
      -r /tmp/repository-requirements.txt \
      -r /tmp/requirements.txt \
      cmake==4.4.2 ninja==1.13.0 \
    && holoscan version
