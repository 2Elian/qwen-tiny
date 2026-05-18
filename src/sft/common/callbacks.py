"""Training visualization callback — logs curves and generates plots."""

import csv
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import TrainerCallback


class TrainingVisualCallback(TrainerCallback):
    """
    Collects train/eval loss, learning rate, grad norm, and speed during training.
    Saves individual plots and a combined dashboard to output_dir/plots/ on exit.
    """

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.grad_norms = []
        self.train_speeds = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.train_losses.append((step, logs["loss"]))
        if "eval_loss" in logs:
            self.eval_losses.append((step, logs["eval_loss"]))
        if "learning_rate" in logs:
            self.learning_rates.append((step, logs["learning_rate"]))
        if "grad_norm" in logs:
            self.grad_norms.append((step, logs["grad_norm"]))
        if "train_samples_per_second" in logs:
            self.train_speeds.append((step, logs["train_samples_per_second"]))

    def on_train_end(self, args, state, control, **kwargs):
        output_dir = Path(args.output_dir)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        self._save_single_plots(plots_dir)
        self._save_dashboard(plots_dir)
        self._save_metrics_csv(plots_dir)
        print(f"[Plots] saved to {plots_dir}")

    # ── internal helpers ──────────────────────────────────────────

    def _save_single_plots(self, plots_dir):
        if self.train_losses:
            steps, vals = zip(*self.train_losses)
            self._plot(steps, vals, "Train Loss", "blue", plots_dir / "train_loss.png")
        if self.eval_losses:
            steps, vals = zip(*self.eval_losses)
            self._plot(steps, vals, "Eval Loss", "red", plots_dir / "eval_loss.png")
        if self.learning_rates:
            steps, vals = zip(*self.learning_rates)
            self._plot(steps, vals, "Learning Rate", "green", plots_dir / "learning_rate.png")
        if self.grad_norms:
            steps, vals = zip(*self.grad_norms)
            self._plot(steps, vals, "Gradient Norm", "magenta", plots_dir / "grad_norm.png")
        if self.train_speeds:
            steps, vals = zip(*self.train_speeds)
            self._plot(steps, vals, "Train Speed", "cyan", plots_dir / "train_speed.png")

    def _save_dashboard(self, plots_dir):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Training Dashboard", fontsize=14, fontweight="bold")
        if self.train_losses:
            s, v = zip(*self.train_losses)
            axes[0, 0].plot(s, v, "b-", linewidth=1)
            axes[0, 0].set_title("Train Loss"); axes[0, 0].grid(True, alpha=0.3)
        if self.eval_losses:
            s, v = zip(*self.eval_losses)
            axes[0, 1].plot(s, v, "r-o", linewidth=1.5, markersize=3)
            axes[0, 1].set_title("Eval Loss"); axes[0, 1].grid(True, alpha=0.3)
        if self.train_losses and self.eval_losses:
            ts, tv = zip(*self.train_losses); es, ev = zip(*self.eval_losses)
            axes[0, 2].plot(ts, tv, "b-", linewidth=1, alpha=0.7, label="train")
            axes[0, 2].plot(es, ev, "r-o", linewidth=1.5, markersize=3, label="eval")
            axes[0, 2].set_title("Train vs Eval"); axes[0, 2].legend(fontsize=8)
            axes[0, 2].grid(True, alpha=0.3)
        if self.learning_rates:
            s, v = zip(*self.learning_rates)
            axes[1, 0].plot(s, v, "g-", linewidth=1)
            axes[1, 0].set_title("Learning Rate"); axes[1, 0].grid(True, alpha=0.3)
        if self.grad_norms:
            s, v = zip(*self.grad_norms)
            axes[1, 1].plot(s, v, "m-", linewidth=1, alpha=0.7)
            if len(v) > 10:
                w = min(20, len(v) // 5)
                r = np.convolve(v, np.ones(w) / w, mode="valid")
                axes[1, 1].plot(s[w - 1:], r, "k--", linewidth=2)
            axes[1, 1].set_title("Gradient Norm"); axes[1, 1].grid(True, alpha=0.3)
        if self.train_speeds:
            s, v = zip(*self.train_speeds)
            axes[1, 2].plot(s, v, "c-", linewidth=1)
            axes[1, 2].set_title("Train Speed (samples/sec)"); axes[1, 2].grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(plots_dir / "dashboard.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_metrics_csv(self, plots_dir):
        step_map = {}
        for s, v in self.train_losses:
            step_map.setdefault(s, {})["train_loss"] = v
        for s, v in self.eval_losses:
            step_map.setdefault(s, {})["eval_loss"] = v
        for s, v in self.learning_rates:
            step_map.setdefault(s, {})["learning_rate"] = v
        for s, v in self.grad_norms:
            step_map.setdefault(s, {})["grad_norm"] = v
        for s, v in self.train_speeds:
            step_map.setdefault(s, {})["train_speed"] = v
        all_steps = sorted(step_map.keys())
        with open(plots_dir / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["step", "train_loss", "eval_loss", "learning_rate", "grad_norm", "train_speed"]
            )
            writer.writeheader()
            for s in all_steps:
                row = {"step": s}; row.update(step_map[s]); writer.writerow(row)

    @staticmethod
    def _plot(x, y, title, color, path):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y, color=color, linewidth=1.5)
        ax.set_title(title); ax.set_xlabel("Step"); ax.grid(True, alpha=0.3)
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
