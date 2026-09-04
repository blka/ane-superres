#!/usr/bin/env python3
"""Konwersja SRVGGNetCompact (Real-ESRGAN pth) → CoreML mlpackage.

Użycie:
  convert.py PTH OUT.mlpackage WxH [--upscale 4]

Wymaga: torch, coremltools (setup.sh --torch).
Architektura: SRVGGNetCompact (realesr-animevideov3, realesrnet-x4plus), standalone.
"""
import sys

import torch
import torch.nn.functional as F
from torch import nn
import coremltools as ct


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super().__init__()
        self.num_in_ch, self.num_out_ch = num_in_ch, num_out_ch
        self.num_feat, self.num_conv = num_feat, num_conv
        self.upscale, self.act_type = upscale, act_type

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(self._act())
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._act())
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def _act(self):
        if self.act_type == 'relu':
            return nn.ReLU(inplace=True)
        if self.act_type == 'leakyrelu':
            return nn.LeakyReLU(0.1, inplace=True)
        return nn.PReLU(num_parameters=self.num_feat)

    def forward(self, x):
        out = x
        for m in self.body:
            out = m(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode='nearest')


def load_pth(path):
    sd = torch.load(path, map_location='cpu')
    for k in ('params_ema', 'params'):
        if k in sd:
            sd = sd[k]
    n_conv = (max(int(k.split('.')[1]) for k in sd if k.startswith('body.')) - 1) // 2
    upscale = int(sd[[k for k in sd if k.endswith('.weight')][-1]].shape[0] / 3) ** 0.5
    m = SRVGGNetCompact(num_conv=n_conv, upscale=int(upscale))
    m.load_state_dict(sd)
    return m.eval()


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    pth, out, wh = sys.argv[1], sys.argv[2], sys.argv[3].lower()
    w, h = (int(v) for v in wh.split('x'))
    model = load_pth(pth)
    traced = torch.jit.trace(model, (torch.rand(1, 3, h, w),))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name='input', shape=(1, 3, h, w))],
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.save(out)
    print(f'zapisano {out} (wejście {w}x{h})')


if __name__ == '__main__':
    main()