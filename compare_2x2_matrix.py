#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.expanduser("~/ros2_ws/evaluation_results")

CONDITIONS = [
    ("injection_haptic_no_force_aug", "1. Haptic Demo\nGeen Force Input", "#1f77b4"),
    ("injection_no_haptic_no_force_aug", "2. No-Haptic Demo\nGeen Force Input", "#ff7f0e"),
    ("injection_haptic_with_force_aug", "3. Haptic Demo\nMET Force Input", "#2ca02c"),
    ("injection_no_haptic_with_force_aug", "4. No-Haptic Demo\nMET Force Input", "#d62728"),
]

def main():
    stats = []

    print("\n" + "=" * 85)
    print(f"{'CONDITIE (2x2 MATRIX)':<32} | {'SUCCES %':<10} | {'TIJD TOT SUCCES':<18} | {'PIEKKRACHT (N)':<15}")
    print("-" * 85)

    for folder, label, color in CONDITIONS:
        csv_path = os.path.join(BASE_DIR, folder, "summary_results.csv")
        clean_label = label.replace("\n", " + ")
        if not os.path.exists(csv_path):
            print(f"{clean_label:<32} | {'Geen data gevonden':<45}")
            stats.append({
                "label": label, "color": color, "success_rate": 0.0,
                "mean_time": 0.0, "std_time": 0.0, "mean_force": 0.0, "std_force": 0.0, "n_total": 0
            })
            continue

        df = pd.read_csv(csv_path)
        n_total = len(df)
        df_succ = df[df["outcome"] == "SUCCESS"]
        n_succ = len(df_succ)

        success_rate = (n_succ / n_total * 100.0) if n_total > 0 else 0.0
        mean_time = df_succ["active_time_s"].mean() if n_succ > 0 else 0.0
        std_time = df_succ["active_time_s"].std() if n_succ > 1 else 0.0
        mean_force = df["peak_force_n"].mean() if n_total > 0 else 0.0
        std_force = df["peak_force_n"].std() if n_total > 1 else 0.0

        print(
            f"{clean_label:<32} | {success_rate:5.1f}% ({n_succ}/{n_total}) | "
            f"{mean_time:5.2f} ± {std_time:4.2f} s     | {mean_force:5.2f} ± {std_force:4.2f} N"
        )

        stats.append({
            "label": label,
            "color": color,
            "success_rate": success_rate,
            "mean_time": mean_time,
            "std_time": std_time,
            "mean_force": mean_force,
            "std_force": std_force,
            "n_total": n_total
        })

    print("=" * 85 + "\n")

    # Genereer het overkoepelende 2x2 vergelijkingsdashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=140)
    fig.suptitle("2x2 Matrix Evaluatie: Haptic Feedback vs. Force Input tijdens Inference", fontsize=14, fontweight="bold")

    labels = [s["label"] for s in stats]
    colors = [s["color"] for s in stats]
    x = np.arange(len(labels))

    # 1. Success Rate (%)
    ax1 = axes[0, 0]
    sr = [s["success_rate"] for s in stats]
    bars1 = ax1.bar(x, sr, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax1.set_title("1. Nauwkeurigheid / Success Rate (%)")
    ax1.set_ylabel("Succes (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, 115)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar, val in zip(bars1, sr):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{val:.1f}%", ha="center", fontweight="bold")

    # 2. Gemiddelde Tijd tot Succes (s)
    ax2 = axes[0, 1]
    mt = [s["mean_time"] for s in stats]
    st = [s["std_time"] for s in stats]
    bars2 = ax2.bar(x, mt, yerr=st, capsize=6, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax2.set_title("2. Gemiddelde Zuivere Tijd tot Succes (s)")
    ax2.set_ylabel("Tijd (s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar, val in zip(bars2, mt):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.2f}s", ha="center")

    # 3. Gemiddelde Piekkracht (N)
    ax3 = axes[1, 0]
    mf = [s["mean_force"] for s in stats]
    sf = [s["std_force"] for s in stats]
    bars3 = ax3.bar(x, mf, yerr=sf, capsize=6, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax3.set_title("3. Gemiddelde Piekkracht tijdens Prikken (N)")
    ax3.set_ylabel("Piekkracht ||F|| (N)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar, val in zip(bars3, mf):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.2f}N", ha="center")

    # 4. Visuele 2x2 Matrix Heatmap van de Success Rate
    ax4 = axes[1, 1]
    matrix_data = np.array([
        [stats[0]["success_rate"], stats[2]["success_rate"]],  # Rij 0: Met Haptic Demo
        [stats[1]["success_rate"], stats[3]["success_rate"]],  # Rij 1: Zonder Haptic Demo
    ])
    im = ax4.imshow(matrix_data, cmap="YlGn", vmin=0, vmax=100)
    ax4.set_title("4. 2x2 Matrix Overzicht (Success Rate %)")
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(["Geen Force Input\n(Policy A)", "MET Force Input\n(Policy B)"], fontsize=10, fontweight="bold")
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(["Met Haptic\nFeedback", "Zonder Haptic\nFeedback"], fontsize=10, fontweight="bold")

    for i in range(2):
        for j in range(2):
            idx = (0 if i == 0 else 1) + (0 if j == 0 else 2)
            text_str = f"{matrix_data[i, j]:.1f}%\n({stats[idx]['mean_time']:.1f}s | {stats[idx]['mean_force']:.1f}N)"
            ax4.text(j, i, text_str, ha="center", va="center", color="black", fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04, label="Success Rate (%)")
    plt.tight_layout(rect=[0, 0.01, 1, 0.94])

    out_plot = os.path.join(BASE_DIR, "2x2_matrix_eindvergelijking_injectie_Final.png")
    plt.savefig(out_plot)
    print(f"✅ Eindvergelijkingsplot opgeslagen in: {out_plot}")

if __name__ == "__main__":
    main()