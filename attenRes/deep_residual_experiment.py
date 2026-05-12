#!/usr/bin/env python3
"""
Deep Network: Residual vs Non-Residual Comparison Experiment
============================================================

Compares deep MLPs (10 / 30 / 50 / 100 layers) with and without residual
connections, across multiple activation functions (ReLU, GELU, Tanh, SiLU).

Metrics tracked per layer:
  - Forward activation: mean, std, norm
  - Gradient flowing into each layer: norm
  - Weight gradient norm

Outputs:
  - figures/residual_experiment/   ← all plots (PNG, 200 dpi)
  - figures/residual_experiment/results.json  ← numerical summary

Usage:
    python deep_residual_experiment.py
    python deep_residual_experiment.py --no-train  # skip training, only init stats
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# matplotlib setup
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Global style
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.figsize": (10, 6),
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIDDEN_DIM = 128
INPUT_DIM = 128
OUTPUT_DIM = 128
BATCH_SIZE = 256
TRAIN_STEPS = 500
LR = 1e-3
DEPTHS = [10, 30, 50, 100]
ACTIVATIONS = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "silu": nn.SiLU(),
}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "figures", "residual_experiment")

# ---------------------------------------------------------------------------
# Network definitions
# ---------------------------------------------------------------------------

class DeepMLP(nn.Module):
    """Stack of Linear -> Activation layers, with optional residual skip."""

    def __init__(self, num_layers: int, hidden_dim: int, activation: nn.Module,
                 use_residual: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        self.input_proj = nn.Linear(INPUT_DIM, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                activation.__class__() if hasattr(activation, '__class__') else activation,
            )
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, OUTPUT_DIM)
        self._init_weights()

    def _init_weights(self):
        # Kaiming init for all Linear layers first
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in",
                                         nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # For residual networks, scale hidden layer weights so that
        # Var(act(Wx)) << Var(x) at init.  Without this, x = x + act(Wx)
        # causes variance to grow exponentially with depth for ReLU-type
        # activations.  Scaling by 1/sqrt(L) is the Fixup-style approach.
        if self.use_residual:
            scale = 1.0 / math.sqrt(self.num_layers)
            for blk in self.layers:
                lin = blk[0]  # nn.Linear
                lin.weight.data.mul_(scale)
                if lin.bias is not None:
                    lin.bias.data.mul_(scale)

        # Last projection: small init to start near output ≈ 0
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            out = layer(x)
            if self.use_residual:
                x = x + out
            else:
                x = out
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# Hook-based metrics collection
# ---------------------------------------------------------------------------

def attach_hooks(model: DeepMLP):
    """Attach forward hooks to capture post-activation values and tensor-level
    gradient hooks for every sub-layer.

    Hook placement:
      - input_proj  (Linear, no activation)
      - For each hidden layer: output of Sequential(Linear, Activation)
      - output_proj (Linear, no activation)

    Forward values are captured via ``register_forward_hook``.
    Gradients are captured via ``Tensor.register_hook`` on the live output tensor.
    """
    n_hidden = model.num_layers
    n_total = 1 + n_hidden + 1  # input_proj + hidden layers + output_proj
    fwd_records: list[dict] = [{} for _ in range(n_total)]
    bwd_records: list[dict] = [{} for _ in range(n_total)]

    def _capture_and_hook(idx, out):
        """Record forward stats and attach a gradient hook on the live tensor."""
        o = out.detach()
        fwd_records[idx] = {
            "mean": float(o.mean().item()),
            "std": float(o.std().item()),
            "norm": float(o.norm().item()),
        }

        def grad_hook(grad):
            g = grad.detach()
            bwd_records[idx] = {
                "input_grad_norm": float(g.norm().item()),
                "input_grad_mean": float(g.mean().item()),
                "input_grad_std": float(g.std().item()),
            }

        out.register_hook(grad_hook)

    # ---- input_proj ----
    def input_proj_hook(module, inp, out):
        _capture_and_hook(0, out)

    model.input_proj.register_forward_hook(input_proj_hook)

    # ---- hidden layers (hook on Sequential output = post-activation) ----
    def make_hidden_hook(idx):
        def hook(module, inp, out):
            _capture_and_hook(idx, out)

        return hook

    for i, blk in enumerate(model.layers):
        blk.register_forward_hook(make_hidden_hook(i + 1))

    # ---- output_proj ----
    def output_proj_hook(module, inp, out):
        _capture_and_hook(n_hidden + 1, out)

    model.output_proj.register_forward_hook(output_proj_hook)

    return fwd_records, bwd_records


def collect_weight_grads(model: DeepMLP):
    """Return list of weight gradient norms per Linear layer."""
    records = []
    for m in [model.input_proj] + [b[0] for b in model.layers] + [model.output_proj]:
        if m.weight.grad is not None:
            records.append(float(m.weight.grad.norm().item()))
        else:
            records.append(0.0)
    return records


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def make_synthetic_data(n_samples: int = 2000):
    """Generate a simple regression task: map random vectors through a fixed
    shallow non-linear function so the target has structure but is learnable."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, INPUT_DIM)

    # Fixed teacher: 2-layer MLP with tanh
    with torch.no_grad():
        w1 = torch.randn(INPUT_DIM, 64) * 0.5
        w2 = torch.randn(64, OUTPUT_DIM) * 0.5
        Y = torch.tanh(X @ w1) @ w2
        Y = Y + 0.05 * torch.randn_like(Y)  # light noise

    # Train / test split
    n_train = int(n_samples * 0.8)
    return (X[:n_train], Y[:n_train]), (X[n_train:], Y[n_train:])


# ---------------------------------------------------------------------------
# Single forward + backward pass for diagnostics
# ---------------------------------------------------------------------------

def diagnostic_pass(model: DeepMLP, x: torch.Tensor, y: torch.Tensor,
                    loss_fn: nn.Module):
    """Run one fwd+bwd pass and return (loss, fwd_records, bwd_records, wgrad_norms)."""
    fwd_records, bwd_records = attach_hooks(model)

    model.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()

    wgrad_norms = collect_weight_grads(model)
    return float(loss.item()), fwd_records, bwd_records, wgrad_norms


# ---------------------------------------------------------------------------
# Training loop (short)
# ---------------------------------------------------------------------------

def train_model(model: DeepMLP, train_data, test_data, steps: int, lr: float):
    """Train for `steps` and return loss history."""
    X_train, Y_train = train_data
    X_test, Y_test = test_data
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {"train_loss": [], "test_loss": [], "step": []}

    model.train()
    n_samples = X_train.shape[0]

    for step in range(steps):
        # Mini-batch
        idx = torch.randint(0, n_samples, (BATCH_SIZE,))
        xb, yb = X_train[idx], Y_train[idx]

        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                test_loss = float(loss_fn(model(X_test), Y_test).item())
            history["step"].append(step)
            history["train_loss"].append(float(loss.item()))
            history["test_loss"].append(test_loss)
            model.train()

    return history


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def make_figure_dir():
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR


def plot_forward_norms(results: dict, depth: int, out_dir: str):
    """For a given depth, plot forward activation norm vs layer index.
    One panel per activation function; two curves (w/ and w/o residual)."""
    n_acts = len(ACTIVATIONS)
    fig, axes = plt.subplots(1, n_acts, figsize=(4.5 * n_acts, 4), sharey=False)
    if n_acts == 1:
        axes = [axes]

    for ax, (act_name, _act_fn) in zip(axes, ACTIVATIONS.items()):
        for res_label, res_flag in [("Residual", True), ("Plain", False)]:
            key = (depth, act_name, res_flag)
            if key not in results:
                continue
            fwd = results[key].get("fwd_norms", [])
            if not fwd:
                continue
            layers = np.arange(len(fwd))
            ax.plot(layers, fwd, marker=".", markersize=3, linewidth=1.2,
                    label=res_label, alpha=0.85)
        ax.set_title(f"{act_name.upper()} (L={depth})")
        ax.set_xlabel("Layer")
        ax.set_yscale("log")
        ax.legend()
    fig.suptitle(f"Forward Activation Norm per Layer  (depth={depth})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"forward_norms_depth{depth}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_gradient_norms(results: dict, depth: int, out_dir: str):
    """Gradient norm flowing into each layer."""
    n_acts = len(ACTIVATIONS)
    fig, axes = plt.subplots(1, n_acts, figsize=(4.5 * n_acts, 4), sharey=False)
    if n_acts == 1:
        axes = [axes]

    for ax, (act_name, _act_fn) in zip(axes, ACTIVATIONS.items()):
        for res_label, res_flag in [("Residual", True), ("Plain", False)]:
            key = (depth, act_name, res_flag)
            if key not in results:
                continue
            bwd = results[key].get("bwd_norms", [])
            if not bwd:
                continue
            layers = np.arange(len(bwd))
            ax.plot(layers, bwd, marker=".", markersize=3, linewidth=1.2,
                    label=res_label, alpha=0.85)
        ax.set_title(f"{act_name.upper()} (L={depth})")
        ax.set_xlabel("Layer")
        ax.set_yscale("log")
        ax.legend()
    fig.suptitle(f"Input Gradient Norm per Layer  (depth={depth})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"gradient_norms_depth{depth}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_weight_gradient_norms(results: dict, depth: int, out_dir: str):
    """Weight gradient norm per layer."""
    n_acts = len(ACTIVATIONS)
    fig, axes = plt.subplots(1, n_acts, figsize=(4.5 * n_acts, 4), sharey=False)
    if n_acts == 1:
        axes = [axes]

    for ax, (act_name, _act_fn) in zip(axes, ACTIVATIONS.items()):
        for res_label, res_flag in [("Residual", True), ("Plain", False)]:
            key = (depth, act_name, res_flag)
            if key not in results:
                continue
            wg = results[key].get("weight_grad_norms", [])
            if not wg:
                continue
            layers = np.arange(len(wg))
            ax.plot(layers, wg, marker=".", markersize=3, linewidth=1.2,
                    label=res_label, alpha=0.85)
        ax.set_title(f"{act_name.upper()} (L={depth})")
        ax.set_xlabel("Layer")
        ax.set_yscale("log")
        ax.legend()
    fig.suptitle(f"Weight Gradient Norm per Layer  (depth={depth})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"weight_grad_norms_depth{depth}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_training_curves(training_results: dict, depth: int, out_dir: str):
    """Test loss curves for all configurations at a given depth."""
    n_acts = len(ACTIVATIONS)
    fig, axes = plt.subplots(1, n_acts, figsize=(4.5 * n_acts, 4), sharey=False)
    if n_acts == 1:
        axes = [axes]

    for ax, (act_name, _act_fn) in zip(axes, ACTIVATIONS.items()):
        for res_label, res_flag in [("Residual", True), ("Plain", False)]:
            key = (depth, act_name, res_flag)
            if key not in training_results:
                continue
            hist = training_results[key]
            ax.plot(hist["step"], hist["test_loss"], linewidth=1.2,
                    label=res_label, alpha=0.85)
        ax.set_title(f"{act_name.upper()} (L={depth})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Test MSE")
        ax.set_yscale("log")
        ax.legend()
    fig.suptitle(f"Test Loss during Training  (depth={depth})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"training_curves_depth{depth}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_summary_heatmap(results: dict, out_dir: str):
    """Summary heatmap: gradient norm ratio (last layer / first layer) across
    all (depth, activation, residual) combinations."""
    rows = []
    for depth in DEPTHS:
        for act_name in ACTIVATIONS:
            for res_flag in [True, False]:
                key = (depth, act_name, res_flag)
                if key not in results:
                    continue
                bwd = results[key].get("bwd_norms", [])
                if len(bwd) < 2:
                    continue
                # ratio of last to first gradient norm (small ratio = vanishing)
                ratio = bwd[-1] / max(bwd[0], 1e-12)
                rows.append({
                    "depth": depth,
                    "activation": act_name,
                    "residual": "Yes" if res_flag else "No",
                    "grad_ratio_last_first": ratio,
                    "first_grad_norm": bwd[0],
                    "last_grad_norm": bwd[-1],
                    "forward_norm_first": results[key].get("fwd_norms", [0])[0],
                    "forward_norm_last": results[key].get("fwd_norms", [0, 0])[-1] if len(results[key].get("fwd_norms", [])) > 1 else 0,
                })

    # Build a text table
    fig, ax = plt.subplots(figsize=(14, len(rows) * 0.4 + 2))
    ax.axis("off")

    col_labels = ["Depth", "Activation", "Residual",
                  "Grad Ratio (last/first)", "First Grad", "Last Grad"]
    cell_text = []
    for r in rows:
        cell_text.append([
            str(r["depth"]),
            r["activation"].upper(),
            r["residual"],
            f'{r["grad_ratio_last_first"]:.4f}',
            f'{r["first_grad_norm"]:.4f}',
            f'{r["last_grad_norm"]:.4f}',
        ])

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    # Color-code the ratio column
    for i, r in enumerate(rows):
        ratio = r["grad_ratio_last_first"]
        # green if ratio is healthy (>0.1), yellow if medium, red if vanishing
        if ratio > 0.1:
            color = "#c8e6c9"  # light green
        elif ratio > 0.01:
            color = "#fff9c4"  # light yellow
        else:
            color = "#ffcdd2"  # light red
        for j in range(6):
            table[(i + 1, j)].set_facecolor(color)

    ax.set_title("Gradient Flow Summary: Ratio of Last-to-First Layer Input Gradient Norm\n"
                 "Green = healthy flow | Yellow = moderate | Red = vanishing",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "summary_gradient_ratio.png")
    fig.savefig(path)
    plt.close(fig)
    return path, rows


def plot_cross_depth_comparison(results: dict, out_dir: str):
    """Plot gradient norm ratio vs depth, one line per (activation, residual) combo."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for act_name in ACTIVATIONS:
        for res_label, res_flag, ls in [("Residual", True, "-"), ("Plain", False, "--")]:
            ratios = []
            depths_present = []
            for depth in DEPTHS:
                key = (depth, act_name, res_flag)
                if key not in results:
                    continue
                bwd = results[key].get("bwd_norms", [])
                if len(bwd) < 2:
                    continue
                ratio = bwd[-1] / max(bwd[0], 1e-12)
                ratios.append(ratio)
                depths_present.append(depth)
            if ratios:
                ax.plot(depths_present, ratios, marker="o", markersize=6,
                        linestyle=ls, linewidth=1.5,
                        label=f"{act_name.upper()} {res_label}", alpha=0.85)

    ax.set_xlabel("Network Depth")
    ax.set_ylabel("Gradient Norm Ratio  (last layer / first layer)")
    ax.set_yscale("log")
    ax.set_title("Gradient Flow vs Network Depth\n"
                 "(ratio < 1 = vanishing, ratio ≈ 1 = healthy flow)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "cross_depth_gradient_ratio.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_forward_distribution(results: dict, out_dir: str):
    """Box-plot style: forward activation mean/std per layer for selected configs."""
    # Focus on depth=50 and depth=100, ReLU
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, depth in enumerate([50, 100]):
        for col, act_name in enumerate(["relu", "gelu"]):
            ax = axes[row][col]
            for res_label, res_flag, color in [("Residual", True, "#2196F3"),
                                                ("Plain", False, "#F44336")]:
                key = (depth, act_name, res_flag)
                if key not in results:
                    continue
                fwd = results[key].get("fwd_norms", [])
                if not fwd:
                    continue
                layers = np.arange(len(fwd))
                ax.fill_between(layers, 0, fwd, alpha=0.3, color=color,
                                label=res_label)
                ax.plot(layers, fwd, color=color, linewidth=1.0, alpha=0.8)
            ax.set_title(f"{act_name.upper()}  depth={depth}")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Forward Norm")
            ax.set_yscale("log")
            ax.legend()
    fig.suptitle("Forward Activation Norm Envelope  (ReLU & GELU, depth 50/100)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "forward_envelope.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Deep residual vs non-residual experiment")
    p.add_argument("--no-train", action="store_true",
                   help="Skip training phase (only init diagnostic)")
    p.add_argument("--train-steps", type=int, default=TRAIN_STEPS,
                   help=f"Training steps (default: {TRAIN_STEPS})")
    p.add_argument("--depths", type=int, nargs="+", default=DEPTHS,
                   help=f"Depths to test (default: {DEPTHS})")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    depths = args.depths
    train_steps = 0 if args.no_train else args.train_steps

    out_dir = make_figure_dir()
    print(f"Output directory: {out_dir}")
    print(f"Device: {device}")
    print(f"Depths: {depths}")
    print(f"Train steps: {train_steps}")

    # Shared dataset
    (X_train, Y_train), (X_test, Y_test) = make_synthetic_data()
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    loss_fn = nn.MSELoss()

    # Storage
    # results[(depth, act_name, use_residual)] = {
    #     "fwd_norms": [...], "bwd_norms": [...], "weight_grad_norms": [...]
    # }
    results: dict = {}
    training_results: dict = {}

    total_configs = len(depths) * len(ACTIVATIONS) * 2  # ×2 for residual flag
    count = 0

    for depth in depths:
        for act_name, act_fn in ACTIVATIONS.items():
            for use_residual in [True, False]:
                count += 1
                res_str = "Residual" if use_residual else "Plain"
                print(f"\n[{count}/{total_configs}] "
                      f"Depth={depth:3d}  Act={act_name.upper():5s}  "
                      f"Mode={res_str}")

                torch.manual_seed(42)  # same init seed for fair comparison

                model = DeepMLP(
                    num_layers=depth,
                    hidden_dim=HIDDEN_DIM,
                    activation=act_fn,
                    use_residual=use_residual,
                ).to(device)

                # ---- diagnostic pass (on a fixed batch) ----
                x_diag = X_train[:BATCH_SIZE]
                y_diag = Y_train[:BATCH_SIZE]
                loss_val, fwd_recs, bwd_recs, wgrad_norms = diagnostic_pass(
                    model, x_diag, y_diag, loss_fn)

                fwd_norms = [r.get("norm", 0.0) for r in fwd_recs if r]
                bwd_norms = [r.get("input_grad_norm", 0.0) for r in bwd_recs if r]

                key = (depth, act_name, use_residual)
                results[key] = {
                    "fwd_norms": fwd_norms,
                    "bwd_norms": bwd_norms,
                    "weight_grad_norms": wgrad_norms,
                    "init_loss": loss_val,
                    "num_layers": depth,
                    "activation": act_name,
                    "residual": use_residual,
                }

                n_fwd = len(fwd_norms)
                n_bwd = len(bwd_norms)
                print(f"    Init loss: {loss_val:.4f}  |  "
                      f"Fwd norm: [{fwd_norms[0]:.2e}, ..., {fwd_norms[-1]:.2e}]  "
                      f"({n_fwd} layers)  |  "
                      f"Grad norm: [{bwd_norms[0]:.2e}, ..., {bwd_norms[-1]:.2e}]  "
                      f"({n_bwd} layers)")

                # ---- training ----
                if train_steps > 0:
                    torch.manual_seed(42)
                    model2 = DeepMLP(
                        num_layers=depth,
                        hidden_dim=HIDDEN_DIM,
                        activation=act_fn,
                        use_residual=use_residual,
                    ).to(device)
                    history = train_model(model2, (X_train, Y_train),
                                          (X_test, Y_test), train_steps, LR)
                    training_results[key] = history
                    print(f"    Final test loss: {history['test_loss'][-1]:.6f}")

    # -------------------------------------------------------------------
    # Save numerical results
    # -------------------------------------------------------------------
    serializable = {}
    for key, val in results.items():
        depth, act_name, res_flag = key
        skey = f"L{depth}_{act_name}_{'res' if res_flag else 'plain'}"
        serializable[skey] = {
            "depth": depth,
            "activation": act_name,
            "residual": res_flag,
            "fwd_norms": val["fwd_norms"],
            "bwd_norms": val["bwd_norms"],
            "weight_grad_norms": val["weight_grad_norms"],
            "init_loss": val["init_loss"],
        }

    # Also save training results
    train_serializable = {}
    for key, hist in training_results.items():
        depth, act_name, res_flag = key
        skey = f"L{depth}_{act_name}_{'res' if res_flag else 'plain'}"
        train_serializable[skey] = hist

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nNumerical results saved → {results_path}")

    if training_results:
        train_results_path = os.path.join(out_dir, "training_results.json")
        with open(train_results_path, "w") as f:
            json.dump(train_serializable, f, indent=2)
        print(f"Training results saved → {train_results_path}")

    # -------------------------------------------------------------------
    # Generate plots
    # -------------------------------------------------------------------
    print("\nGenerating plots ...")

    saved_plots = []

    for depth in depths:
        p = plot_forward_norms(results, depth, out_dir)
        saved_plots.append(p)

        p = plot_gradient_norms(results, depth, out_dir)
        saved_plots.append(p)

        p = plot_weight_gradient_norms(results, depth, out_dir)
        saved_plots.append(p)

        if training_results:
            p = plot_training_curves(training_results, depth, out_dir)
            saved_plots.append(p)

    p, summary_rows = plot_summary_heatmap(results, out_dir)
    saved_plots.append(p)

    p = plot_cross_depth_comparison(results, out_dir)
    saved_plots.append(p)

    p = plot_forward_distribution(results, out_dir)
    saved_plots.append(p)

    print(f"\nAll plots saved to: {out_dir}/")
    for p in saved_plots:
        print(f"  {os.path.basename(p)}")

    # -------------------------------------------------------------------
    # Print final summary table
    # -------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUMMARY: Gradient Norm Ratio (last layer / first layer)")
    print("  Ratio < 0.01  → severe vanishing")
    print("  Ratio 0.01-0.1 → moderate vanishing")
    print("  Ratio > 0.1   → healthy flow")
    print("=" * 90)
    header = f"{'Depth':>6s}  {'Act':>6s}  {'Mode':>10s}  {'Ratio':>8s}  {'FirstGrad':>12s}  {'LastGrad':>12s}"
    print(header)
    print("-" * 90)
    for r in summary_rows:
        print(f"{r['depth']:6d}  {r['activation']:>6s}  {r['residual']:>10s}  "
              f"{r['grad_ratio_last_first']:8.4f}  {r['first_grad_norm']:12.4e}  "
              f"{r['last_grad_norm']:12.4e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
