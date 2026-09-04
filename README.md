# ane-superres

Super-rozdzielczość wideo (SR) na Apple Silicon przez **CoreML / Neural Engine** —
bez PyTorch w runtime, bez Vulkan/ncnn.

Model: `realesr-animevideov3` (SRVGGNetCompact, ×4) z [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN),
skonwertowany do CoreML fp16. Zmierzony throughput (MacBook Air M4, klatka 720×576):

| jednostka | ms/klatkę | film 326k klatek |
|---|---|---|
| **ANE** | **103** | ~9,3 h |
| GPU (Metal) | 146 | ~13 h |
| CPU | 246 | ~22 h |
| ncnn + MoltenVK | 3300 | ~300 h |

## Użycie

```zsh
./setup.sh                  # raz: venv + coremltools + numpy
./sr-upscale VIDEO          # -> VIDEO_sr.mp4 (2× wejścia)
./sr-upscale VIDEO -o OUT.mp4 --target 1080x1440   # dowolny cel
./sr-upscale VIDEO -s 1:35:05 -d 30                # fragment
./sr-upscale VIDEO --no-deint                      # źródło bez przeplotu
```

Pipeline: `ffmpeg` (bwdif + hqdn3d dla źródeł SD) → `sr_driver.py` (ANE ×4)
→ `ffmpeg` (bicubic ½ + ewent. crop do celu) → `h264_videotoolbox`, audio `-c:a copy`.
Całość strumieniowa — brak plików tymczasowych, zużycie dysku = sam wynik.

Dla źródła 720×576 (PAL 4:3) `--target 1080x1440` daje 1440×1080 (crop 36 px).

## Własny model

```zsh
./setup.sh --torch
./convert.py realesrgan-x4plus.pth models/x4plus.mlpackage 720x576
./sr-upscale VIDEO --model models/x4plus.mlpackage
```

Konwertowane z `TensorType` (NCHW float 0–1) — wejście i wyjście modelu to float
0–1, driver skaluje sam. Stałe kształty (ANE wymaga) — jeden model = jeden rozmiar
wejścia; inny rozmiar → nowa konwersja przez `convert.py`.

## Pułapki (nauczone)

- binary `realesrgan-ncnn-vulkan` z release segfaultuje na macOS 27 (`readlink("/proc/self/exe")` — Linux-only; fix: `_NSGetExecutablePath`).
- ncnn nie włącza `VK_KHR_portability_enumeration` → MoltenVK niewidoczny; sterownik Mesa/KosmicKrisp na Apple GPU jest ~4× wolniejszy od MoltenVK.
- coremltools nie wspiera torch `upsample_bicubic2d` — bicubic downscale zewnętrznie (ffmpeg).
- wyjście modelu CoreML nazywa się `var_312` (nie `output`).
- `animevideov3-x2` z ncnn to ten sam model ×4 + bicubic ½ w grafie.