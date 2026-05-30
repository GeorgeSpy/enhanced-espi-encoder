import torch
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description="Audit spectral/Fourier-like keys in a hierarchical checkpoint.")
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path(os.environ["HIERARCHICAL_CKPT_PATH"]) if os.environ.get("HIERARCHICAL_CKPT_PATH") else None,
    help="Path to the hierarchical checkpoint. Defaults to HIERARCHICAL_CKPT_PATH if set.",
)
parser.add_argument(
    "--out",
    type=Path,
    default=Path(r"reports\lefft_ablation_v001\HIERARCHICAL_CKPT_SPECTRAL_KEYS.txt"),
    help="Output report path.",
)
args = parser.parse_args()
if args.checkpoint is None:
    raise SystemExit("Provide --checkpoint or set HIERARCHICAL_CKPT_PATH.")

ckpt_path = args.checkpoint
out_path = args.out
out_path.parent.mkdir(parents=True, exist_ok=True)

try:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

    keys = sorted(state.keys())
    spectral_keys = [
        k for k in keys
        if any(s in k.lower() for s in ["leftp", "lefft", "leftp", "fourier", "fft", "spectral"])
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Total keys: {len(keys)}\n")
        f.write(f"Spectral/Fourier-like keys: {len(spectral_keys)}\n\n")
        for k in spectral_keys:
            try:
                shape = tuple(state[k].shape)
            except Exception:
                shape = "NA"
            f.write(f"{k} | shape={shape}\n")

    print(f"[DONE] wrote {out_path}")
    print(f"Spectral-like keys: {len(spectral_keys)}")
except Exception as e:
    print(f"[ERROR] Could not load checkpoint or compute keys: {e}")
    # Write empty / error report
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"ERROR: {e}\n")
