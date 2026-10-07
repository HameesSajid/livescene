"""Does half precision (FP16) work on this GPU, and is it faster?

Some GPUs (GTX 16-series among them) are slower or numerically unreliable in FP16.
This script measures it instead of assuming: it runs the depth model in FP32 and
under FP16 autocast on the same frame and reports speed and whether the FP16 output
is finite and close to FP32.

Run: python tools/precision_check.py path/to/video.mp4
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import livescene.paths  # noqa: E402,F401  (sets HF_HOME)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from livescene.perception import MetricDepth  # noqa: E402


def bench(fn, n=30):
    for _ in range(5):
        out = fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0, out


def main():
    ok, frame = cv2.VideoCapture(sys.argv[1]).read()
    assert ok, "cannot read video"
    frame = cv2.resize(frame, (960, 540))
    depth = MetricDepth()

    ms32, d32 = bench(lambda: depth(frame))

    def half():
        with torch.autocast("cuda", dtype=torch.float16):
            return depth(frame)

    ms16, d16 = bench(half)
    finite = float(np.isfinite(d16).mean())
    err = float(np.nanmean(np.abs(d16 - d32) / d32)) if finite > 0 else float("nan")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"depth FP32: {ms32:.1f} ms")
    print(f"depth FP16: {ms16:.1f} ms, finite pixels {finite:.1%}, mean relative difference vs FP32 {err:.2%}")


if __name__ == "__main__":
    main()
