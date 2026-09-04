# Benchmarks

All numbers measured on a **MacBook Air M4** (macOS 27, no thermal throttling,
wall power). Workload: one frame **720×576 → 2880×2304** (×4), model
`realesr-animevideov3` (SRVGGNetCompact, 16 blocks, 2.5 MB fp16 weights).

Steady-state ms per frame, inference only (no ffmpeg pipe, no encode):

| backend | ms/frame | × slower than ANE |
|---|---:|---:|
| CoreML, compute units = **ANE** | **103** | 1× |
| CoreML, compute units = GPU | 146 | 1.4× |
| CoreML, compute units = CPU | 246 | 2.4× |
| ncnn + MoltenVK (patched) | 3300 | 32× |
| ncnn + Mesa KosmicKrisp | ~240000 | ~2300× |
| ncnn + Mesa llvmpipe (CPU) | 38000 | ~370× |

Notes per backend:

- **ANE** — the winner, but only because the model is tiny and compiled with fixed
  shapes. fp16, `ComputeUnit.ALL` resolves to ANE here. Measured inside the full
  pipe (ffmpeg → driver → ffmpeg), i.e. including rawvideo I/O overhead.
- **GPU (CoreML/Metal)** — respectable but slower than ANE on this workload.
  Selecting `ComputeUnit.CPU_AND_NE` vs explicit GPU changes dispatch; measure
  per model.
- **ncnn + MoltenVK** — after fixing the two macOS bugs (see
  [ncnn-notes.md](ncnn-notes.md)), MoltenVK is visible and functional, but compute
  shader throughput on Apple GPUs via MoltenVK is ~30× below ANE for this model.
- **Mesa KosmicKrisp** — Apple's GPU driver in Mesa. Catastrophic for ncnn compute
  shaders: x4plus at ~4 minutes per frame. Slower than pure-CPU llvmpipe.
  Do not use; if you must run Vulkan, force MoltenVK ICD.
- **llvmpipe** — software rasterizer, 38 s/frame. Included as the "at least it
  finishes" baseline.

## Film-level estimates

6531 s PAL source (163k frames), `bwdif=send_field` → 326k frames at 50 fps:

| backend | total render |
|---|---:|
| ANE | ~9.3 h |
| GPU | ~13 h |
| CPU | ~22 h |
| ncnn + MoltenVK | ~300 h |

Plus encode time (h264_videotoolbox is negligible next to inference).

## Method

```zsh
# CoreML: sr_driver.py logs ms/frame at SR_LOG_EVERY intervals, steady state
SR_LOG_EVERY=100 ./sr_driver.py models/av3x4_tensor.mlpackage 720x576 < frames.raw

# ncnn: timed batch of 250 frames
time ./realesrgan-ncnn-vulkan -i frames -o out -n realesr-animevideov3 -s 4 -g <dev>
```

GPU device IDs under MoltenVK/Mesa were resolved with `vulkaninfo --summary`.
Full-render throughput (103 ms/frame) is what `sr-upscale` actually delivers:
end-to-end pipe including deinterlace, encode and disk I/O.