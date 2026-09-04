# Pipeline internals

## Chain

```
ffmpeg -vf "bwdif=mode=send_field,hqdn3d,format=rgb24" -f rawvideo
  → sr_driver.py  (CoreML fp16, ANE, model input WxH → output 4W×4H)
  → ffmpeg -f rawvideo -s 4Wx4H
      -vf "scale=2W:2H:flags=bicubic,crop=TH:TW:dx:dy,setsar=1"
      -c:v h264_videotoolbox  -map "1:a?" -c:a copy -shortest
```

Three stages, all joined by pipes. Peak memory is one frame per stage.

## Why the model is ×4 only

`realesr-animevideov3` ships with `scale=2` and `scale=4` weight files, and ncnn
treats them as separate models. They are not: the `-x2` variant is the **same
×4 model** followed by a bicubic ×0.5 resize inside the graph (`Interp` layer,
`Resize 0=3`). coremltools cannot convert torch's `upsample_bicubic2d` anyway,
so the honest equivalent is:

- run the ×4 model,
- do the bicubic ×0.5 **in ffmpeg** (`scale=2W:2H:flags=bicubic`),
- crop to the exact target size from the center.

This reproduces ncnn's `x2` output bit-exactly (verified: max pixel diff 0).

## Target-size logic in `sr-upscale`

Let input be `W×H`, model output `4W×4H`, target `TW×TH`:

1. `TW×TH == 4W×4H` → pass through, fix SAR (`setsar=1`).
2. `TW ≤ 2W and TH ≤ 2H` → bicubic ½ to `2W×2H`, then centered crop to target.
3. otherwise → bicubic direct scale to target.

Case 2 is the common one (e.g. PAL 4:3: 720×576 → bicubic 1440×1152 → crop
1440×1080, dropping 36 px top and bottom — exactly the ncnn `x2` + 1080p crop
behavior).

## Deinterlace frame-rate trap

`bwdif=mode=send_field` emits one frame **per field**: a 25 fps interlaced PAL
source becomes 50 frames/s out of the first ffmpeg. The rawvideo pipe has no
timestamps, so the second ffmpeg's `-r` must be doubled too, or the encoder
dups/drops frames. `sr-upscale` parses `r_frame_rate` and doubles it when
deinterlacing (`25/1` → `50/1`).

## CoreML conversion choices

- **`TensorType`, not `ImageType`**: `ImageType` inputs accept only PIL images —
  useless in a raw pipe. Tensor I/O is NCHW float 0–1; the driver does
  `/255` and `*255` around the model.
- **fp16 (`compute_precision=ct.precision.FLOAT16`)**: ANE is an fp16 machine;
  fp32 falls off the ANE fast path.
- **fixed shapes**: the ANE requires static shapes — `(1,3,576,720)` compiled
  once, reused for every frame.
- **`.mlpackage`, not `.mlmodel`**: mlprogram models require the package format.
- **output name is auto-generated** (`var_312` in our case): `sr_driver.py` reads
  input/output names from `model.get_io_spec()` style introspection instead of
  hardcoding, so converted variants of other models also work.

## Driver details

`sr_driver.py MODEL WxH`:

- reads rawvideo rgb24 on stdin, writes rawvideo rgb24 on stdout;
- feeds frames one at a time (the model has batch 1 fixed);
- `np.clip(y, 0, 1) * 255` on the output — the model can overshoot 0–1, and
  a missing clip shows up as black or wrapped frames;
- `SR_LOG_EVERY=N` logs steady-state `ms/frame` every N frames to stderr,
  prints `DONE n frames: ...` at EOF.

## Model conversion (`convert.py`)

Standalone `SRVGGNetCompact` implementation — no `basicsr` dependency, no
module registry games (upstream `srvgg_arch.py` 404s from Real-ESRGAN releases;
the architecture here is reconstructed from the checkpoint keys). It infers
`num_conv` from key count and `upscale` from the last conv's output size, traces
the module, and converts with fixed shapes. Requires `setup.sh --torch`
(torch is ~2 GB; not needed for inference-only usage).