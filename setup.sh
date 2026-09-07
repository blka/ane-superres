#!/usr/bin/env zsh
# setup.sh — environment for ane-superres (python venv + dependencies).
# Models come pre-converted in models/; for custom conversion: --torch.
set -euo pipefail
DIR="${0:A:h}"

# system prerequisites that cannot come from pip
missing=()
for cmd in ffmpeg ffprobe; do
  command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
done
(( ${#missing[@]} )) && {
  echo "missing: ${missing[*]} — install first: brew install ffmpeg"
  exit 1
}
command -v python3.13 >/dev/null 2>&1 || {
  echo "missing: python3.13 — install first: brew install python@3.13"
  echo "(coremltools has native bindings only up to Python 3.13)"
  exit 1
}

# coremltools has native bindings only for python <= 3.13 (3.14 is broken)
python3.13 -m venv "$DIR/.venv"
"$DIR/.venv/bin/pip" install -q --upgrade pip
"$DIR/.venv/bin/pip" install -q "coremltools>=9" numpy
"$DIR/.venv/bin/python" -c "import coremltools; print('coremltools', coremltools.__version__)"

if [[ "${1:-}" == "--torch" ]]; then
  "$DIR/.venv/bin/pip" install -q torch torchvision
  echo "torch installed (model conversion)"
  if [[ ! -f "$DIR/realesr-animevideov3.pth" ]]; then
    curl -sL -o "$DIR/realesr-animevideov3.pth" \
      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3/realesr-animevideov3.pth"
    echo "weights downloaded: realesr-animevideov3.pth"
  fi
fi

chmod +x "$DIR/sr-upscale" "$DIR/sr_driver.py" 2>/dev/null || true
echo "done — use: $DIR/sr-upscale VIDEO"