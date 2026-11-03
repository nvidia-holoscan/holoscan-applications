# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Development image for the surgical_deidentification application.
#
# Build and run through holoscan-cli from the repository root:
#   $ export HOLOSCAN_CLI_BASE_IMAGE=<holoscan-sdk-dev-image>
#   $ holoscan build surgical_deidentification
#   $ holoscan run surgical_deidentification
#
# Override the SDK base image at build time:
#   $ holoscan build surgical_deidentification \
#        --base-img nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13

# holoscan-cli supplies BASE_IMAGE from --base-img (or HOLOSCAN_CLI_BASE_IMAGE).
# Keeping it unset here prevents an unreviewed SDK image from becoming an implicit default.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# System packages. FFmpeg -dev is a build-time (pkg_check_modules REQUIRED) dependency of the
# video_file_reader / video_file_writer operators. libyaml-cpp-dev is linked by de_id.cpp for
# config parsing (yaml-cpp is a private dep of the SDK's inference library and is not
# re-exported).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        pkg-config \
        libavformat-dev \
        libavcodec-dev \
        libavutil-dev \
        libswscale-dev \
        libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir \
      -r /tmp/requirements.txt \
    && holoscan version
