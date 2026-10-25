# VideoFileWriterOp

Consumes Holoscan `Tensor`s on its input port, converts each incoming frame to a
libav-compatible pixel format on the CPU, encodes it with a libav software encoder, and
muxes the encoded packets into an output container (mp4, mkv, …) written to local disk.

The operator accepts both host- and device-backed input tensors — it uses
`Tensor::copy_to_host(...)` to pull the frame into a host staging buffer regardless of where
the producer allocated the tensor. If the input tensor lives on device memory, this triggers
a per-compute D→H PCIe transfer on the operator's CUDA stream.

## Parameters (summary)

| Parameter          | Effect                                                                          |
|--------------------|---------------------------------------------------------------------------------|
| `path`             | Output file path passed to `avio_open`.                                         |
| `tensor_port_name` | Name of the consumed input port.                                                |
| `codec`            | libav encoder name resolved via `avcodec_find_encoder_by_name` (e.g. `libx264`).|
| `output_container` | Optional muxer name (`mp4`, `mkv`, …). If empty, inferred from `path`.          |
| `input_pix_fmt`    | One of `rgb24`, `bgr24`, `rgb_fp32`, `bgr_fp32`, `grayscale_u8`, `grayscale_fp32`. `*_fp32` triggers the host-side clamp-to-uint8 loop. |
| `input_layout`     | `nchw` or `nhwc`. `nchw` triggers the host-side transpose loop.                 |
| `strip_batch_dim`  | If true, a leading batch dim of 1 is expected on the input.                     |
| `width` / `height` | Encoder frame dimensions (required).                                            |
| `fps`              | Encoder time-base + framerate + GOP size.                                       |
| `bitrate`          | Target encoder bitrate.                                                         |
| `frame_count`      | Cap on how many frames are encoded before subsequent inputs are silently dropped. |
| `expected_dtype` / `expected_shape` | Optional runtime validation against the incoming Tensor.       |
