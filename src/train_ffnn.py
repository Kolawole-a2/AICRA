from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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

    f = np.load(args.features, allow_pickle=True)
    X = f["X"].astype(np.float32)
    y = np.load(args.labels)["y"].astype(np.float32)

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
