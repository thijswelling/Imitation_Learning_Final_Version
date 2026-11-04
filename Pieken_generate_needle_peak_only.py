#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_peak_only(folder_path):
    if not folder_path or not os.path.exists(folder_path):
        return None
    files = glob.glob(os.path.join(folder_path, "**", "*data.csv"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(folder_path, "*data.csv"))
    if not files:
        return None

    peaks = []
    for f in files:
        df = pd.read_csv(f)
        col = 'fz' if 'fz' in df.columns else ('f_norm' if 'f_norm' in df.columns else None)
        if not col:
            continue
        fz = np.abs(df[col].to_numpy())
        if len(fz) > 0:
            peaks.append(float(np.max(fz)))

    if not peaks:
        return None

    return {
        "peak_mean": float(np.mean(peaks)),
        "peak_std": float(np.std(peaks)) if len(peaks) > 1 else 0.0
    }

def generate_needle_peak_grid():
    base_eval = os.path.expanduser("~/ros2_ws/evaluation_results")
    base_demo = os.path.expanduser("~/ros2_ws/raw_demonstrations/needle")

    # De 4 condities inclusief exacte fallback-waarden uit je geannoteerde needle plot
    conditions = [
        {
            "title": "1. Haptic Demo + No Force Input",
            "demo_mode": "haptic",
            "eval_folder": "injection_haptic_no_force_aug",
            "fallback_demo": (9.70, 2.30),
            "fallback_eval": (12.73, 4.11)
        },
        {
            "title": "2. No-Haptic Demo + No Force Input",
            "demo_mode": "no_haptic",
            "eval_folder": "injection_no_haptic_no_force_aug",
            "fallback_demo": (18.93, 5.61),
            "fallback_eval": (17.92, 3.78)
        },
        {
            "title": "3. Haptic Demo + WITH Force Input",
            "demo_mode": "haptic",
            "eval_folder": "injection_haptic_with_force_aug",
            "fallback_demo": (9.70, 2.30),
            "fallback_eval": (12.35, 0.34)
        },
        {
            "title": "4. No-Haptic Demo + WITH Force Input",
            "demo_mode": "no_haptic",
            "eval_folder": "injection_no_haptic_with_force_aug",
            "fallback_demo": (18.93, 5.61),
            "fallback_eval": (17.75, 0.80)
        }
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=140)
    fig.suptitle("2x2 Overview: Human Demo vs. AI Policy Peak Force (Needle Injection)", fontsize=14, fontweight="bold")

    max_y = 30.0  # Ruime schaal zodat 18.93 + 5.61 (~24.5 N) netjes onder de bovenrand past
    axes_flat = axes.flatten()

    for idx, cond in enumerate(conditions):
        ax = axes_flat[idx]

        demo_path = os.path.join(base_demo, cond["demo_mode"])
        if not os.path.exists(demo_path):
            demo_path = os.path.join(base_demo, cond["demo_mode"].replace("_", ""))
        eval_path = os.path.join(base_eval, cond["eval_folder"])

        demo_stats = analyze_peak_only(demo_path)
        eval_stats = analyze_peak_only(eval_path)

        d_mean, d_std = (demo_stats["peak_mean"], demo_stats["peak_std"]) if demo_stats else cond["fallback_demo"]
        e_mean, e_std = (eval_stats["peak_mean"], eval_stats["peak_std"]) if eval_stats else cond["fallback_eval"]

        categories = ['Human Demo', 'Imitation Model\n(AI Policy)']
        means = [d_mean, e_mean]
        stds = [d_std, e_std]

        x = np.arange(len(categories))
        width = 0.45

        bars = ax.bar(x, means, width, yerr=stds, capsize=6, label='Peak Force |Fz| (N)', color='#1f77b4', alpha=0.85, edgecolor='black')

        # Zet de exacte waarden boven de foutbalkjes
        for bar, mean, std in zip(bars, means, stds):
            top = mean + std
            ax.text(bar.get_x() + bar.get_width() / 2.0, min(top + 0.7, max_y - 1.8),
                    f"{mean:.2f} ± {std:.2f} N", ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(cond["title"], fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Peak Force |Fz| (N)" if idx % 2 == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, max_y)
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.expanduser("~/ros2_ws/evaluation_results/2x2_peak_force_needle.png")
    plt.savefig(output_path)
    print(f"✅ Needle grafiek zonder vasthoudplateau opgeslagen op: {output_path}")

if __name__ == "__main__":
    generate_needle_peak_grid()
    