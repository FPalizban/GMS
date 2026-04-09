#!/usr/bin/env python3
"""
Autoencoder training for germline signature representation learning.

- Trains a symmetric deep autoencoder (ReLU+BN+Dropout) on CASE features
- Early stopping on validation MSE
- Exports embeddings for cases (and controls if provided) using the trained encoder
- Saves training curves and UMAP of the latent space

Usage:
  python scripts/autoencoder_train.py \
    --features-cases outputs/features/features_cases.csv \
    --features-controls outputs/features/features_controls.csv \
    --out outputs/ae \
    --latent-dim 32 --hidden-dims 256 128 --dropout 0.10 \
    --batch-size 128 --max-epochs 500 --val-split 0.15 \
    --patience 30 --lr 1.5e-3 --repeats 1
"""

import argparse
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# optional (for quick 2D visualization)
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AE(nn.Module):
    def __init__(self, in_dim: int, hidden: list, latent: int, dropout: float):
        super().__init__()
        h1, h2 = hidden if len(hidden) == 2 else (hidden[0], hidden[-1])
        self.enc = nn.Sequential(
            nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, h2), nn.BatchNorm1d(h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.BatchNorm1d(h1), nn.ReLU(),
            nn.Linear(h1, in_dim),
        )

    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z)
        return xhat, z


def fit_one(X: np.ndarray, args, device: str = "cpu"):
    """Train one AE with early stopping; return (model, history dict)."""
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    val_size = int(len(ds) * args.val_split)
    tr_size = max(1, len(ds) - val_size)
    tr, val = random_split(ds, [tr_size, val_size], generator=torch.Generator().manual_seed(42))

    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = AE(in_dim=X.shape[1], hidden=args.hidden_dims, latent=args.latent_dim, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    best_val, best_state, noimp = float("inf"), None, 0
    hist = {"train": [], "val": []}

    for epoch in range(args.max_epochs):
        # train
        model.train()
        tr_loss_sum = 0.0
        for (xb,) in tr_loader:
            xb = xb.to(device)
            xhat, _ = model(xb)
            loss = crit(xhat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item() * len(xb)
        tr_loss = tr_loss_sum / max(1, tr_size)

        # val
        model.eval()
        va_loss_sum = 0.0
        with torch.no_grad():
            for (xb,) in va_loader:
                xb = xb.to(device)
                xhat, _ = model(xb)
                va_loss_sum += crit(xhat, xb).item() * len(xb)
        va_loss = va_loss_sum / max(1, val_size)

        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)

        # early stopping
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            noimp = 0
        else:
            noimp += 1
            if noimp >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def save_curves(hist, out_png: Path, title: str):
    fig = plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def embed_with(model: AE, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z = model.enc(torch.from_numpy(X.astype(np.float32)).to(device)).cpu().numpy()
    return z


def make_umap(Z: np.ndarray, labels=None, out_png: Path = None):
    if not HAS_UMAP:
        return
    emb = umap.UMAP(random_state=42).fit_transform(Z)
    fig = plt.figure(figsize=(5, 4), dpi=300)
    if labels is None:
        plt.scatter(emb[:, 0], emb[:, 1], s=10)
    else:
        # labels can be None or array-like; for now, just plot uncolored
        plt.scatter(emb[:, 0], emb[:, 1], s=10)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    fig.savefig(out_png if out_png is not None else "umap_latent.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Train AE and export embeddings")
    p.add_argument("--features-cases", required=True, help="CSV from feature_engineering (cases)")
    p.add_argument("--features-controls", required=False, default=None, help="CSV from feature_engineering (controls)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--repeats", type=int, default=1, help="Repeat training to check stability; embeddings from run 1 are saved as canonical")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load features
    X_cases_df = pd.read_csv(args.features_cases, index_col=0)
    X_cases = X_cases_df.values.astype(np.float32)
    case_ids = X_cases_df.index.astype(str).tolist()

    if args.features_controls and Path(args.features_controls).exists():
        X_ctrls_df = pd.read_csv(args.features_controls, index_col=0)
        X_ctrls = X_ctrls_df.values.astype(np.float32)
        ctrl_ids = X_ctrls_df.index.astype(str).tolist()
    else:
        X_ctrls_df, X_ctrls, ctrl_ids = None, None, None

    # train (possibly multiple repeats)
    runs = []
    for r in range(args.repeats):
        model, hist = fit_one(X_cases, args, device=device)
        save_curves(hist, outdir / f"ae_loss_run{r+1}.png", title=f"AE loss (run {r+1})")
        # save model state for each run
        torch.save(model.state_dict(), outdir / f"ae_run{r+1}.pt")
        runs.append({"model_path": str(outdir / f"ae_run{r+1}.pt"), "final_val_mse": float(hist['val'][-1])})

        # only compute embeddings for run 1 as canonical
        if r == 0:
            Z_cases = embed_with(model, X_cases, device=device)
            pd.DataFrame(Z_cases, index=case_ids).to_csv(outdir / "embeddings_cases.csv")

            if X_ctrls is not None:
                Z_ctrls = embed_with(model, X_ctrls, device=device)
                pd.DataFrame(Z_ctrls, index=ctrl_ids).to_csv(outdir / "embeddings_controls.csv")

            # optional UMAP visualization
            make_umap(Z_cases, labels=None, out_png=outdir / "umap_latent_cases.png")

    # manifest
    manifest = {
        "in": {
            "features_cases_csv": str(Path(args.features_cases).resolve()),
            "features_controls_csv": str(Path(args.features_controls).resolve()) if X_ctrls_df is not None else None,
        },
        "model": {
            "latent_dim": args.latent_dim,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "val_split": args.val_split,
            "patience": args.patience,
            "device": device,
            "repeats": args.repeats,
        },
        "runs": runs,
        "out": {
            "model_pt_run1": runs[0]["model_path"] if runs else None,
            "embeddings_cases_csv": str(outdir / "embeddings_cases.csv"),
            "embeddings_controls_csv": str(outdir / "embeddings_controls.csv") if X_ctrls_df is not None else None,
            "loss_curves_png": [str(outdir / f"ae_loss_run{i+1}.png") for i in range(args.repeats)],
            "umap_latent_cases_png": str(outdir / "umap_latent_cases.png") if HAS_UMAP else None,
        },
    }
    with open(outdir / "ae_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n✅ Autoencoder training complete.")
    print(f"   model (run1) → {runs[0]['model_path'] if runs else 'n/a'}")
    print(f"   embeddings (cases) → {outdir / 'embeddings_cases.csv'}")
    if X_ctrls_df is not None:
        print(f"   embeddings (controls) → {outdir / 'embeddings_controls.csv'}")
    print(f"   loss curves → {', '.join([p for p in manifest['out']['loss_curves_png']])}")
    if HAS_UMAP:
        print(f"   UMAP (cases) → {outdir / 'umap_latent_cases.png'}")


if __name__ == "__main__":
    main()
