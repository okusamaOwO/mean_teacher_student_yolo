"""
Inspect YOLOv9/GELAN model to find correct layer indices for P3, P4, P5.

Usage:
    python helpers/inspect_model.py --cfg models/detect/gelan-c.yaml --nc 80
    python helpers/inspect_model.py --cfg models/detect/gelan-s.yaml --nc 1
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from models.yolo import Model


def inspect_model(cfg, nc=80, imgsz=640):
    model = Model(cfg, ch=3, nc=nc)
    model.eval()

    print("=" * 80)
    print(f"Config: {cfg}  |  Layers: {len(model.model)}")
    print("=" * 80)

    # Collect shapes via hooks
    features = {}
    hooks = []

    def make_hook(idx):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                features[idx] = out
        return hook

    for i, layer in enumerate(model.model):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    x = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    # Print all layers
    print(f"\n{'Idx':>4s}  {'Class':30s}  {'Output Shape':20s}  {'Stride':>6s}")
    print("-" * 70)
    for idx in sorted(features.keys()):
        layer = model.model[idx]
        feat = features[idx]
        name = layer.__class__.__name__
        shape_str = str(list(feat.shape))
        stride = imgsz // feat.shape[2] if feat.dim() == 4 else ""
        marker = ""
        if isinstance(stride, int) and stride in (8, 16, 32):
            marker = f"  <-- P{int(torch.log2(torch.tensor(stride / 8.0)).item()) + 3}"
        print(f"[{idx:3d}]  {name:30s}  {shape_str:20s}  {str(stride):>6s}{marker}")

    # Detect head info
    detect_layer = model.model[-1]
    detect_idx = len(model.model) - 1
    print(f"\nDetect head at index [{detect_idx}]: {detect_layer.__class__.__name__}")
    if hasattr(detect_layer, 'f'):
        print(f"  Input from layers (f): {detect_layer.f}")

    # Suggest neck layers
    print("\n" + "=" * 80)
    print("SUGGESTED NECK LAYERS for MGD hooks:")
    print("=" * 80)

    candidates = {}
    for idx in sorted(features.keys()):
        feat = features[idx]
        if feat.dim() == 4:
            s = imgsz // feat.shape[2]
            if s in (8, 16, 32):
                candidates[s] = (idx, feat.shape[1])

    for s in [8, 16, 32]:
        if s in candidates:
            idx, ch = candidates[s]
            level = {8: 'P3', 16: 'P4', 32: 'P5'}[s]
            print(f"  {level}: layer_index={idx}, channels={ch}, stride={s}")

    indices = [candidates[s][0] for s in [8, 16, 32] if s in candidates]
    channels = [candidates[s][1] for s in [8, 16, 32] if s in candidates]
    print(f"\n  layer_indices = {indices}")
    print(f"  channels_list = {channels}")

    return indices, channels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/detect/gelan-c.yaml')
    parser.add_argument('--nc', type=int, default=80)
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()
    inspect_model(args.cfg, args.nc, args.imgsz)
