# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Development image for apriltag_detection: the SDK image plus NVIDIA VPI 4 for
# the vpi backend. The cpu and cuda backends need nothing beyond the SDK.
#
# From the repository root:
#   $ export HOLOSCAN_CLI_BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13
#   $ holoscan build apriltag_detection
#   $ holoscan run apriltag_detection --no-docker-build

# holoscan-cli supplies BASE_IMAGE from --base-img (or HOLOSCAN_CLI_BASE_IMAGE).
# Keeping it unset here prevents an unreviewed SDK image from becoming an implicit default.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# VPI 4 from NVIDIA's Jetson package repository, which publishes it for x86_64
# hosts and for Jetson. This is a third-party apt repository: the build accepts
# its signing key only with the fingerprint below, apt verifies every package
# against that key, and the VPI packages are pinned to one version. (In gpg's
# colon listing the first fingerprint record is the primary key's.)
ARG VPI_REPOSITORY_RELEASE=r38.4
ARG VPI_VERSION=4.0.5
ARG VPI_REPOSITORY_FINGERPRINT=3C6D1FF3100C8C3ABB0869C0E6543461A9996195
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg; \
    case "$(dpkg --print-architecture)" in \
      amd64) suite="x86_64/noble" ;; \
      arm64) suite="common" ;; \
      *) echo "unsupported architecture" >&2; exit 1 ;; \
    esac; \
    export GNUPGHOME="$(mktemp -d)"; \
    key="${GNUPGHOME}/nvidia-jetson.asc"; \
    curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc -o "${key}"; \
    keys="$(gpg --show-keys --with-colons "${key}" | awk -F: '$1 == "pub"' | wc -l)"; \
    fingerprint="$(gpg --show-keys --with-colons "${key}" | awk -F: '$1 ~ /^fp/ { print $10; exit }')"; \
    if [ "${keys}" != 1 ] || [ "${fingerprint}" != "${VPI_REPOSITORY_FINGERPRINT}" ]; then \
      echo "unexpected VPI repository signing key: ${keys} key(s), ${fingerprint}" >&2; exit 1; \
    fi; \
    gpg --dearmor < "${key}" > /usr/share/keyrings/nvidia-jetson.gpg; \
    rm -rf "${GNUPGHOME}"; \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/${suite} ${VPI_REPOSITORY_RELEASE} main" \
      > /etc/apt/sources.list.d/nvidia-vpi.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends "libnvvpi4=${VPI_VERSION}" "vpi4-dev=${VPI_VERSION}"; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir \
      -r /tmp/requirements.txt \
    && holoscan version
