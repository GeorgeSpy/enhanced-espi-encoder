import torch
from pathlib import Path

ckpt_path = Path(r"C:\ESPI\logs\train_v6.2_opt\ckpt_phase2_expert.pt")
out_path = Path(r"reports\lefft_ablation_v001\HIERARCHICAL_CKPT_SPECTRAL_KEYS.txt")
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
