# Initial GPU validation

Date: 2026-10-09. Single NVIDIA H20, CUDA 12.8, PyTorch 2.10.0, no FA3 or compilation. The checkpoint SHA-256 matches the pinned manifest. All 1,589 learned tensor keys and shapes were checked, with no missing learned tensors.

| Check | Result |
|---|---|
| Single-image segmentation | Passed; 1.60 s inference after model load; 5.63 GiB peak allocated |
| Three-frame video propagation | Passed; 1.51 s inference after model load; 7.38 GiB peak allocated |

The image test found one target. The three consecutive driving frames retained object ID 0. These are installation smoke tests, not throughput benchmarks or proof of robust tracking, replacement quality, occlusion handling or full-video performance. User media and diagnostic overlays remain private and are not published.
