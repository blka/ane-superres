#!/usr/bin/env zsh
# setup.sh — środowisko dla ane-superres (python venv + zależności).
# Modele są już skonwertowane w models/; do własnej konwersji: --torch.
set -euo pipefail
DIR="${0:A:h}"

# coremltools ma natywne bindingi tylko dla pythona <= 3.13 (3.14 nie działa)
python3.13 -m venv "$DIR/.venv"
"$DIR/.venv/bin/pip" install -q --upgrade pip
"$DIR/.venv/bin/pip" install -q "coremltools>=9" numpy
"$DIR/.venv/bin/python" -c "import coremltools; print('coremltools', coremltools.__version__)"

if [[ "${1:-}" == "--torch" ]]; then
  "$DIR/.venv/bin/pip" install -q torch torchvision
  echo "torch zainstalowany (konwersja modeli)"
fi

chmod +x "$DIR/sr-upscale" "$DIR/sr_driver.py" 2>/dev/null || true
echo "gotowe — używaj: $DIR/sr-upscale VIDEO"