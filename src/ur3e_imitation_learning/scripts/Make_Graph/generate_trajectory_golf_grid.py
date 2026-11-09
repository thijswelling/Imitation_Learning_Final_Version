#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_folder(folder_path):
    if not folder_path or not os.path.exists(folder_path):
        print(f"⚠️ Map niet gevonden: {folder_path}")
        return None
    files = glob.glob(os.path.join(folder_path, "**", "*data.csv"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(folder_path, "*data.csv"))
    if not files:
        print(f"⚠️ Geen CSV bestanden in map: {folder_path}")
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

def generate_trajectory_grid():
    base_eval = os.path.expanduser("~/ros2_ws/evaluation_results")
    base_demo = os.path.expanduser("~/ros2_ws/raw_demonstrations")

    # We koppelen nu expliciet de juiste haptic en no_haptic demomappen per conditie
    conditions = [
        {
            "title": "1. Haptic Demo + No Force Input",
            "demo_folder": "trajectory_golf/haptic", # Pas dit pad aan indien je map anders heet
            "eval_folder": "trajectory_golf_haptic_no_force"
        },
        {
            "title": "2. No-Haptic Demo + No Force Input",
            "demo_folder": "trajectory_golf/no_haptic", # Of trajectory_golf/nohaptic
            "eval_folder": "trajectory_golf_no_haptic_no_force"
        },
        {
            "title": "3. Haptic Demo + WITH Force Input",
            "demo_folder": "trajectory_golf/haptic",
            "eval_folder": "trajectory_golf_haptic_with_force"
        },
        {
            "title": "4. No-Haptic Demo + WITH Force Input",
            "demo_folder": "trajectory_golf/no_haptic",
            "eval_folder": "trajectory_golf_no_haptic_with_force"
        }
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=140)
    fig.suptitle("2x2 Overview: Human Demo vs. AI Policy Force Metrics (Trajectory Golf)", fontsize=14, fontweight="bold")

    max_y = 25.0  # Y-as ruim genoeg voor alle pieken
    axes_flat = axes.flatten()

    for idx, cond in enumerate(conditions):
        ax = axes_flat[idx]
        
        # Zoek het juiste demopad (probeer ook varianten zoals nohaptic zonder underscore)
        d_path = os.path.join(base_demo, cond["demo_folder"])
        if not os.path.exists(d_path):
            d_path = os.path.join(base_demo, cond["demo_folder"].replace("no_haptic", "nohaptic"))
        if not os.path.exists(d_path):
            # Probeer alternatieve hoofdmap
            d_path = os.path.expanduser(f"~/ros2_ws/raw_demonstrations/trajectory/{cond['demo_folder'].split('/')[-1]}")

        eval_path = os.path.join(base_eval, cond["eval_folder"])

        print(f"[{cond['title']}]")
        print(f"  -> Demo map: {d_path}")
        print(f"  -> Eval map: {eval_path}")

        demo_stats = analyze_folder(d_path)
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

        ax.set_title(cond["title"], fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Force |Fz| (N)" if idx % 2 == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, max_y)
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.expanduser("~/ros2_ws/evaluation_results/2x2_comparison_grid_trajectory_golf_final.png")
    plt.savefig(output_path)
    print(f"\n✅ Perfecte grafiek opgeslagen op: {output_path}")

if __name__ == "__main__":
    generate_trajectory_grid()  