#!/usr/bin/env python3
"""CoreML super-resolution: rawvideo rgb24 (stdin) -> ANE model -> rawvideo rgb24 (stdout).

Usage (piped with ffmpeg):
  ffmpeg ... -f rawvideo -pix_fmt rgb24 - | sr_driver.py MODEL WxH | ffmpeg ...

MODEL — an .mlpackage directory with TensorType input (1,3,H,W).
WxH   — input frame size, e.g. 720x576.

Environment: SR_LOG_EVERY (log progress to stderr every N frames, default 500).
"""
import os
import sys
import time

import numpy as np
import coremltools as ct


def main():
    model_dir = sys.argv[1]
    w, h = (int(v) for v in sys.argv[2].lower().split('x'))

    m = ct.models.MLModel(model_dir, compute_units=ct.ComputeUnit.ALL)
    io = list(m.input_description), list(m.output_description)
    in_key, out_key = io[0][0], io[1][0]

    frame_bytes = w * h * 3

    n = 0
    pred_t = 0.0
    t_start = time.perf_counter()
    buf = b''
    stdout = sys.stdout.buffer
    stdin = sys.stdin.buffer
    while True:
        while len(buf) < frame_bytes:
            chunk = stdin.read(frame_bytes - len(buf))
            if not chunk:
                break
            buf += chunk
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf[:frame_bytes], dtype=np.uint8).reshape(h, w, 3)
        buf = b''

        t0 = time.perf_counter()
        x = np.ascontiguousarray(frame.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        y = np.asarray(m.predict({in_key: x})[out_key], dtype=np.float32)  # (1,3,H*s,W*s)
        out = np.clip(y[0], 0, 1) * 255
        out = np.ascontiguousarray(out.transpose(1, 2, 0)).astype(np.uint8)
        if n == 0:
            print(f'model: in {w}x{h} -> out {out.shape[1]}x{out.shape[0]}', file=sys.stderr)
        stdout.write(out.tobytes())
        pred_t += time.perf_counter() - t0
        n += 1

        every = int(os.environ.get('SR_LOG_EVERY', '500'))
        if n % every == 0:
            el = time.perf_counter() - t_start
            print(f'frame {n}: {el / n * 1000:.0f} ms/frame (pred {pred_t / n * 1000:.0f} ms)', file=sys.stderr)

    el = time.perf_counter() - t_start
    if n:
        print(f'DONE {n} frames: {el / n * 1000:.0f} ms/frame (pred {pred_t / n * 1000:.0f} ms)', file=sys.stderr)


if __name__ == '__main__':
    main()