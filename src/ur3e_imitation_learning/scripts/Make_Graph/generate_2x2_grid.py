#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, "**", "*data.csv"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(folder_path, "*data.csv"))
    if not files:
        return None

    peaks, plateaus = [], []
    for f in files:
        df = pd.read_csv(f)
        col = 'fz' if 'fz' in df.columns else ('f_norm' if 'f_norm' in df.columns else None)
        if not col:
            continue
        fz = np.abs(df[col].to_numpy())
        if len(fz) == 0:
            continue
        peaks.append(float(np.max(fz)))
        plateau = fz[int(len(fz) * 0.7):]
        if len(plateau) > 0:
            plateaus.append(float(np.mean(plateau)))

    if not peaks:
        return None

    return {
        "peak_mean": np.mean(peaks),
        "peak_std": np.std(peaks) if len(peaks) > 1 else 0.0,
        "plateau_mean": np.mean(plateaus) if plateaus else 0.0,
        "plateau_std": np.std(plateaus) if len(plateaus) > 1 else 0.0
    }

def generate_grid(use_case="needle"):
    base_eval = os.path.expanduser("~/ros2_ws/evaluation_results")
    base_demo = os.path.expanduser(f"~/ros2_ws/raw_demonstrations/{use_case}")

    # Definieer de 4 condities op basis van je werkelijke mapnamen
    if use_case == "needle":
        conditions = [
            {"name": "1. Haptic Demo\nNo Force Input", "demo_mode": "haptic", "eval_folder": "injection_haptic_no_force_aug"},
            {"name": "2. No-Haptic Demo\nNo Force Input", "demo_mode": "no_haptic", "eval_folder": "injection_no_haptic_no_force_aug"},
            {"name": "3. Haptic Demo\nWITH Force Input", "demo_mode": "haptic", "eval_folder": "injection_haptic_with_force_aug"},
            {"name": "4. No-Haptic Demo\nWITH Force Input", "demo_mode": "no_haptic", "eval_folder": "injection_no_haptic_with_force_aug"}
        ]
    else:
        conditions = [
            {"name": "1. Haptic Demo\nNo Force Input", "demo_mode": "haptic", "eval_folder": "trajectory_haptic_no_force"},
            {"name": "2. No-Haptic Demo\nNo Force Input", "demo_mode": "no_haptic", "eval_folder": "trajectory_nohaptic_no_force"},
            {"name": "3. Haptic Demo\nWITH Force Input", "demo_mode": "haptic", "eval_folder": "trajectory_haptic_with_force"},
            {"name": "4. No-Haptic Demo\nWITH Force Input", "demo_mode": "no_haptic", "eval_folder": "trajectory_nohaptic_with_force"}
        ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=140)
    fig.suptitle(f"2x2 Overview: Human Demo vs. AI Policy Force Metrics ({use_case.capitalize()})", fontsize=15, fontweight="bold")

    max_y = 35.0 if use_case == "needle" else 10.0
    axes_flat = axes.flatten()

    for idx, cond in enumerate(conditions):
        ax = axes_flat[idx]
        demo_path = os.path.join(base_demo, cond["demo_mode"])
        eval_path = os.path.join(base_eval, cond["eval_folder"])

        demo_stats = analyze_folder(demo_path)
        eval_stats = analyze_folder(eval_path)

        categories = ['Human Demo', 'AI Policy']
        p_means = [demo_stats["peak_mean"] if demo_stats else 0, eval_stats["peak_mean"] if eval_stats else 0]
        p_stds = [demo_stats["peak_std"] if demo_stats else 0, eval_stats["peak_std"] if eval_stats else 0]
        pl_means = [demo_stats["plateau_mean"] if demo_stats else 0, eval_stats["plateau_mean"] if eval_stats else 0]
        pl_stds = [demo_stats["plateau_std"] if demo_stats else 0, eval_stats["plateau_std"] if eval_stats else 0]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width/2, p_means, width, yerr=p_stds, capsize=4, label='Peak Force (Peak Fz)', color='#1f77b4', alpha=0.85, edgecolor='black')
        ax.bar(x + width/2, pl_means, width, yerr=pl_stds, capsize=4, label='Plateau Force (Plateau Fz)', color='#2ca02c', alpha=0.85, edgecolor='black')

        ax.set_title(cond["name"], fontsize=11, fontweight="bold")
        ax.set_ylabel("Force |Fz| (N)" if idx % 2 == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, max_y)
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.expanduser(f"~/ros2_ws/evaluation_results/2x2_comparison_grid_{use_case}.png")
    plt.savefig(output_path)
    print(f"✅ 2x2 grid opgeslagen voor {use_case} op: {output_path}")

if __name__ == "__main__":
    generate_grid("needle")
    generate_grid("trajectory")