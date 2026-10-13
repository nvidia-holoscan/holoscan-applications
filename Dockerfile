# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# holoscan-cli supplies BASE_IMAGE from --base-img (or HOLOSCAN_CLI_BASE_IMAGE).
# Keeping it unset here prevents an unreviewed SDK image from becoming an implicit default.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir \
      -r /tmp/requirements.txt \
    && holoscan version
