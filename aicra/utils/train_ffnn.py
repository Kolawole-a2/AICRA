from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


def is_trusted_path(path: Path) -> bool:
    """
    Check if file path is within trusted directories.
    
    Security: Prevents loading arbitrary files from untrusted locations
    that could contain malicious pickle data.
    """
    abs_path = path.resolve()
    trusted_dirs = [
        Path.cwd() / "data",
        Path.cwd() / "artifacts",
        Path.cwd() / "results",
        Path.cwd() / "models",
    ]
    return any(abs_path.is_relative_to(trusted.resolve()) for trusted in trusted_dirs)


def safe_load_npz(path: Path, required_keys: Optional[list[str]] = None) -> dict:
    """
    Safely load .npz file without allow_pickle.
    
    Args:
        path: Path to .npz file
        required_keys: List of required keys in .npz file
    
    Returns:
        Dictionary with loaded arrays
    
    Raises:
        ValueError: If path is not trusted or file structure is invalid
    """
    if not is_trusted_path(path):
        raise ValueError(
            f"File path must be within trusted directories: "
            f"{[str(Path.cwd() / d) for d in ['data', 'artifacts', 'results', 'models']]}"
        )
    
    try:
        data = np.load(path, allow_pickle=False)
        if isinstance(data, np.ndarray):
            raise ValueError(f"Expected .npz file with keys, got .npy array: {path}")
        
        result = {}
        for key in data.keys():
            result[key] = data[key]
        
        if required_keys:
            missing = set(required_keys) - set(result.keys())
            if missing:
                raise ValueError(f"Missing required keys in {path}: {missing}")
        
        return result
    except (KeyError, TypeError, OSError) as e:
        raise ValueError(f"Invalid .npz file structure in {path}: {e}")


class SmallFFNN(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--outdir", default="artifacts")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Safely load features
    feat_path = Path(args.features)
    f = safe_load_npz(feat_path, required_keys=["X"])
    X = f["X"].astype(np.float32)
    
    # Safely load labels
    label_path = Path(args.labels)
    label_data = safe_load_npz(label_path, required_keys=["y"])
    y = label_data["y"].astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallFFNN(X.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device).view(-1, 1)

    model.train()
    for _ in range(10):
        opt.zero_grad()
        preds = model(X_t)
        loss = loss_fn(preds, y_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        probs = model(X_t).cpu().numpy().ravel()
    np.savez(outdir / "ffnn_predictions.npz", probs=probs, labels=y)
    print("Saved FFNN predictions")


if __name__ == "__main__":
    main()
