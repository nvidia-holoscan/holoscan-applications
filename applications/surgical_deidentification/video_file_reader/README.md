# VideoFileReaderOp

Reads a video file (or URL) with FFmpeg, decodes frames on the CPU, converts them to a
uniform pixel format and layout on the CPU, then emits each frame as a Holoscan `Tensor` on
the operator's declared output port.

Emit destination is controlled by `memory_kind`:

- `cuda_device` / `cuda_managed` — every frame is copied from the CPU staging buffer to the
  destination device buffer with a per-tick `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, ...)`
  on the operator's CUDA stream.
- `host` / `pinned_host` — the CPU staging buffer is `memcpy`'d directly into the loaned
  tensor; no CUDA work is performed.

## Where work happens

**All decoding and pixel conversion is on the CPU.** The GPU is only touched by the H→D
upload at the end of `compute()`, and only when `memory_kind` selects device or managed
memory. There is no CUDA-accelerated decode path (NVDEC / `AV_HWACCEL_*`) in this operator —
libav uses its default software codec.

## Parameters (summary)

| Parameter          | Effect                                                                         |
|--------------------|--------------------------------------------------------------------------------|
| `path`             | Local file path or URL passed to `avformat_open_input`.                        |
| `tensor_port_name` | Name of the emitted output port.                                               |
| `output_width` / `output_height` | Optional resize target for `sws_scale`. If unset, native resolution is used. |
| `output_pix_fmt`   | One of `rgb24`, `bgr24`, `rgb_fp32`, `bgr_fp32`, `grayscale_u8`, `grayscale_fp32`. `*_fp32` triggers the host-side normalization loop. |
| `output_layout`    | `nchw` or `nhwc`. `nchw` triggers the host-side transpose loop.               |
| `prepend_batch_dim`| If true, output shape becomes `[1, ...]`.                                     |
| `memory_kind`      | `cuda_device` / `cuda_managed` / `host` / `pinned_host`. Controls whether the H→D copy fires. |
| `frame_count`      | Cap on how many frames are emitted. Any value `≤ 0` means unlimited (the default is `0`); the cap only fires when the parameter is strictly positive. |
| `frame_rate`       | If `> 0`, sleep-based pacing between emissions.                               |
| `loop`             | If true, seek back to the beginning at EOS.                                   |
| `stream_index`     | `-1` picks the best video stream via `av_find_best_stream`.                   |
