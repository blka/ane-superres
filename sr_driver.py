#!/usr/bin/env python3
"""CoreML super-resolution: rawvideo rgb24 (stdin) -> ANE model -> rawvideo rgb24 (stdout).

Usage (piped with ffmpeg):
  ffmpeg ... -f rawvideo -pix_fmt rgb24 - | sr_driver.py MODEL WxH | ffmpeg ...

MODEL — an .mlpackage directory with TensorType input (N,3,H,W). The batch
size N is read from the model; frames are processed N at a time (the final
partial batch is padded by repeating the last frame). Works with batch-1
models too.
WxH   — input frame size, e.g. 720x576.

Inference and postprocess (float->uint8) overlap: the main thread feeds the
model while a worker thread converts finished batches and writes them to
stdout, in input order.

Environment: SR_LOG_EVERY (log progress to stderr every N frames, default 500).
"""
import os
import queue
import sys
import threading
import time

import numpy as np
import coremltools as ct


def main():
    model_dir = sys.argv[1]
    w, h = (int(v) for v in sys.argv[2].lower().split('x'))

    m = ct.models.MLModel(model_dir, compute_units=ct.ComputeUnit.ALL)
    spec = m.get_spec()
    in_key = spec.description.input[0].name
    out_key = spec.description.output[0].name
    model_shape = list(spec.description.input[0].type.multiArrayType.shape)
    if len(model_shape) != 4 or model_shape[1] != 3 or model_shape[2] != h or model_shape[3] != w:
        print(f'model input shape {model_shape} does not match {w}x{h}', file=sys.stderr)
        sys.exit(2)
    batch = model_shape[0]

    frame_bytes = w * h * 3

    q = queue.Queue(maxsize=2)
    write_err = []

    def postprocess():
        # converts finished predicts to rgb24 bytes and writes them in order
        while True:
            item = q.get()
            if item is None:
                return
            try:
                y = item  # (batch,3,H*s,W*s) float32
                out = np.clip(y * 255, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
                stdout.write(np.ascontiguousarray(out).tobytes())
            except Exception as e:  # keep the pipe alive; surface via DONE rc
                write_err.append(e)
                return
            finally:
                q.task_done()

    stdout = sys.stdout.buffer
    stdin = sys.stdin.buffer
    worker = threading.Thread(target=postprocess, daemon=True)
    worker.start()

    n = 0
    pred_t = 0.0
    t_start = time.perf_counter()
    buf = b''
    pending = []  # input frames (h,w,3) not yet sent to the model
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
        pending.append(frame)

        if len(pending) < batch:
            continue

        t0 = time.perf_counter()
        x = np.ascontiguousarray(
            np.stack(pending).transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
        y = np.asarray(m.predict({in_key: x})[out_key], dtype=np.float32)  # (n,3,H*s,W*s)
        q.put(y)
        pred_t += time.perf_counter() - t0
        n += len(pending)
        pending = []

        every = int(os.environ.get('SR_LOG_EVERY', '500'))
        if n % every < batch:
            el = time.perf_counter() - t_start
            print(f'frame {n}: {el / n * 1000:.0f} ms/frame (pred {pred_t / n * 1000:.0f} ms)', file=sys.stderr)

    # tail: pad the last partial batch by repeating the final frame, then
    # emit only the real frames
    if pending:
        k = len(pending)
        last = pending[-1]
        while len(pending) < batch:
            pending.append(last)
        t0 = time.perf_counter()
        x = np.ascontiguousarray(
            np.stack(pending).transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
        y = np.asarray(m.predict({in_key: x})[out_key], dtype=np.float32)
        q.put(y[:k])  # emit only the real frames
        pred_t += time.perf_counter() - t0
        n += k

    q.put(None)
    worker.join()
    el = time.perf_counter() - t_start
    if write_err:
        print(f'ERROR: {write_err[0]}', file=sys.stderr)
        sys.exit(3)
    if n:
        print(f'DONE {n} frames: {el / n * 1000:.0f} ms/frame '
              f'(pred {pred_t / n * 1000:.0f} ms, batch {batch})', file=sys.stderr)


if __name__ == '__main__':
    main()