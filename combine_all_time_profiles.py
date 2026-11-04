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
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        col = 'fz' if 'fz' in df.columns else ('f_norm' if 'f_norm' in df.columns else None)
        if not col:
            continue
        fz = np.abs(df[col].to_numpy(dtype=float))
        if len(fz) > 0:
            peaks.append(float(np.max(fz)))

    if not peaks:
        return None

    return {
        "peak_mean": float(np.mean(peaks)),
        "peak_std": float(np.std(peaks)) if len(peaks) > 1 else 0.0
    }

def load_lerobot_episodes(base_folder, target_len=300):
    if not base_folder or not os.path.exists(base_folder):
        return []
    files = glob.glob(os.path.join(base_folder, "**", "*data.csv"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(base_folder, "**", "*.csv"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(base_folder, "**", "*.parquet"), recursive=True)
    if not files:
        return []

    normalized_runs = []
    for f in files:
        try:
            if f.endswith('.parquet'):
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f)
        except Exception:
            continue

        col = next((c for c in ['fz', 'f_norm', 'observation.state.fz', 'action.fz'] if c in df.columns), None)
        if not col:
            matching_cols = [c for c in df.columns if 'fz' in c.lower() or 'force' in c.lower()]
            if matching_cols:
                col = matching_cols[0]
            else:
                continue

        fz = np.abs(df[col].to_numpy(dtype=float))
        if len(fz) < 5:
            continue

        original_t = np.linspace(0, 1, len(fz))
        common_t = np.linspace(0, 1, target_len)
        resampled_fz = np.interp(common_t, original_t, fz)
        normalized_runs.append((common_t * 100.0, resampled_fz))

    return normalized_runs

def plot_group(ax, runs, color, label_prefix):
    if not runs:
        return
    for i, (t, fz) in enumerate(runs):
        lbl = f"{label_prefix}" if i == 0 else None
        ax.plot(t, fz, color=color, alpha=0.22, linewidth=1.0, label=lbl)

    all_fz = np.array([r[1] for r in runs])
    common_t = runs[0][0]
    mean_fz = np.mean(all_fz, axis=0)
    std_fz = np.std(all_fz, axis=0) if len(runs) > 1 else np.zeros_like(mean_fz)

    ax.plot(common_t, mean_fz, color=color, linewidth=2.0, label=f"{label_prefix} Mean")
    if len(runs) > 1:
        ax.fill_between(common_t, np.maximum(0, mean_fz - std_fz), mean_fz + std_fz, color=color, alpha=0.15)

def generate_4x4_master_numbered():
    base_eval = os.path.expanduser("~/ros2_ws/evaluation_results")
    base_demo = os.path.expanduser("~/ros2_ws/raw_demonstrations")

    conditions = [
        {
            "name": "1. Haptic Demo\n+ No Force",
            "traj_demo": os.path.join(base_demo, "trajectory/haptic_wave"),
            "traj_eval": os.path.join(base_eval, "trajectory_golf_haptic_no_force"),
            "traj_fb_demo": (3.21, 0.67), "traj_fb_eval": (2.48, 0.28),
            "needle_demo": os.path.join(base_demo, "needle/haptic"),
            "needle_eval": os.path.join(base_eval, "injection_haptic_no_force_aug"),
            "needle_fb_demo": (9.70, 2.30), "needle_fb_eval": (12.73, 4.11)
        },
        {
            "name": "2. No-Haptic Demo\n+ No Force",
            "traj_demo": os.path.join(base_demo, "trajectory/no_haptic_wave"),
            "traj_eval": os.path.join(base_eval, "trajectory_golf_no_haptic_no_force"),
            "traj_fb_demo": (4.19, 0.91), "traj_fb_eval": (15.95, 7.01),
            "needle_demo": os.path.join(base_demo, "needle/no_haptic"),
            "needle_eval": os.path.join(base_eval, "injection_no_haptic_no_force_aug"),
            "needle_fb_demo": (18.93, 5.61), "needle_fb_eval": (17.92, 3.78)
        },
        {
            "name": "3. Haptic Demo\n+ WITH Force",
            "traj_demo": os.path.join(base_demo, "trajectory/haptic_wave"),
            "traj_eval": os.path.join(base_eval, "trajectory_golf_haptic_with_force"),
            "traj_fb_demo": (3.21, 0.67), "traj_fb_eval": (3.68, 0.15),
            "needle_demo": os.path.join(base_demo, "needle/haptic"),
            "needle_eval": os.path.join(base_eval, "injection_haptic_with_force_aug"),
            "needle_fb_demo": (9.70, 2.30), "needle_fb_eval": (12.35, 0.34)
        },
        {
            "name": "4. No-Haptic Demo\n+ WITH Force",
            "traj_demo": os.path.join(base_demo, "trajectory/no_haptic_wave"),
            "traj_eval": os.path.join(base_eval, "trajectory_golf_no_haptic_with_force"),
            "traj_fb_demo": (4.19, 0.91), "traj_fb_eval": (6.28, 0.63),
            "needle_demo": os.path.join(base_demo, "needle/no_haptic"),
            "needle_eval": os.path.join(base_eval, "injection_no_haptic_with_force_aug"),
            "needle_fb_demo": (18.93, 5.61), "needle_fb_eval": (17.75, 0.80)
        }
    ]

    fig, axes = plt.subplots(4, 4, figsize=(21, 17), dpi=140)
    fig.suptitle("Master Overview: 4x4 Numbered Comparative Matrix (Trajectory vs. Needle)", fontsize=16, fontweight="bold", y=0.98)

    col_titles = [
        "Trajectory: Peak Force Bar Chart",
        "Trajectory: Force Profile Over Time",
        "Needle: Peak Force Bar Chart",
        "Needle: Force Profile Over Time"
    ]

    # Letter-nummers voor subplots: (a) t/m (p)
    labels = [
        ['(a)', '(b)', '(c)', '(d)'],
        ['(e)', '(f)', '(g)', '(h)'],
        ['(i)', '(j)', '(k)', '(l)'],
        ['(m)', '(n)', '(o)', '(p)']
    ]

    for row_idx, cond in enumerate(conditions):
        categories = ['Human', 'AI Policy']
        x = np.arange(len(categories))

        # --- Kolom 0: Trajectory Bar Chart ---
        ax_t_bar = axes[row_idx, 0]
        d_stats = analyze_peak_only(cond["traj_demo"])
        e_stats = analyze_peak_only(cond["traj_eval"])
        d_m, d_s = (d_stats["peak_mean"], d_stats["peak_std"]) if d_stats else cond["traj_fb_demo"]
        e_m, e_s = (e_stats["peak_mean"], e_stats["peak_std"]) if e_stats else cond["traj_fb_eval"]
        
        means = [d_m, e_m]
        stds = [d_s, e_s]
        bars = ax_t_bar.bar(x, means, 0.5, yerr=stds, capsize=4, color='#1f77b4', alpha=0.85, edgecolor='black')
        for bar, mean, std in zip(bars, means, stds):
            ax_t_bar.text(bar.get_x() + bar.get_width()/2.0, min(mean + std + 0.4, 23.2), f"{mean:.2f}N", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        ax_t_bar.set_ylim(0, 25.0)
        ax_t_bar.set_xticks(x)
        ax_t_bar.set_xticklabels(categories, fontsize=8)
        ax_t_bar.set_ylabel("Peak Fz (N)", fontsize=8.5)
        ax_t_bar.grid(True, axis='y', linestyle=':', alpha=0.6)
        ax_t_bar.text(0.03, 0.91, labels[row_idx][0], transform=ax_t_bar.transAxes, fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # --- Kolom 1: Trajectory Time Profile ---
        ax_t_time = axes[row_idx, 1]
        t_demo_runs = load_lerobot_episodes(cond["traj_demo"])
        t_eval_runs = load_lerobot_episodes(cond["traj_eval"])
        plot_group(ax_t_time, t_demo_runs, color='#1f77b4', label_prefix='Human')
        plot_group(ax_t_time, t_eval_runs, color='#d62728', label_prefix='AI Policy')
        ax_t_time.set_ylim(0, 25.0)
        ax_t_time.set_xlim(0, 100)
        ax_t_time.set_ylabel("Force Fz (N)", fontsize=8.5)
        ax_t_time.grid(True, linestyle=':', alpha=0.6)
        ax_t_time.text(0.03, 0.91, labels[row_idx][1], transform=ax_t_time.transAxes, fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        if row_idx == 3:
            ax_t_time.set_xlabel("Task Progress (%)", fontsize=8.5)
        if row_idx == 0:
            ax_t_time.legend(loc="upper right", fontsize=7, framealpha=0.9)

        # --- Kolom 2: Needle Bar Chart ---
        ax_n_bar = axes[row_idx, 2]
        nd_stats = analyze_peak_only(cond["needle_demo"])
        ne_stats = analyze_peak_only(cond["needle_eval"])
        nd_m, nd_s = (nd_stats["peak_mean"], nd_stats["peak_std"]) if nd_stats else cond["needle_fb_demo"]
        ne_m, ne_s = (ne_stats["peak_mean"], ne_stats["peak_std"]) if ne_stats else cond["needle_fb_eval"]
        
        n_means = [nd_m, ne_m]
        n_stds = [nd_s, ne_s]
        n_bars = ax_n_bar.bar(x, n_means, 0.5, yerr=n_stds, capsize=4, color='#1f77b4', alpha=0.85, edgecolor='black')
        for bar, mean, std in zip(n_bars, n_means, n_stds):
            ax_n_bar.text(bar.get_x() + bar.get_width()/2.0, min(mean + std + 0.6, 28.2), f"{mean:.2f}N", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        ax_n_bar.set_ylim(0, 30.0)
        ax_n_bar.set_xticks(x)
        ax_n_bar.set_xticklabels(categories, fontsize=8)
        ax_n_bar.set_ylabel("Peak Fz (N)", fontsize=8.5)
        ax_n_bar.grid(True, axis='y', linestyle=':', alpha=0.6)
        ax_n_bar.text(0.03, 0.91, labels[row_idx][2], transform=ax_n_bar.transAxes, fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # --- Kolom 3: Needle Time Profile ---
        ax_n_time = axes[row_idx, 3]
        n_demo_runs = load_lerobot_episodes(cond["needle_demo"])
        n_eval_runs = load_lerobot_episodes(cond["needle_eval"])
        plot_group(ax_n_time, n_demo_runs, color='#1f77b4', label_prefix='Human')
        plot_group(ax_n_time, n_eval_runs, color='#d62728', label_prefix='AI Policy')
        ax_n_time.set_ylim(0, 30.0)
        ax_n_time.set_xlim(0, 100)
        ax_n_time.set_ylabel("Force Fz (N)", fontsize=8.5)
        ax_n_time.grid(True, linestyle=':', alpha=0.6)
        ax_n_time.text(0.03, 0.91, labels[row_idx][3], transform=ax_n_time.transAxes, fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        if row_idx == 3:
            ax_n_time.set_xlabel("Task Progress (%)", fontsize=8.5)

        # Kolom titels bovenste rij
        for col_idx in range(4):
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=10.5, fontweight="bold", pad=10)

        # Rij-labels aan de linkerkant
        axes[row_idx, 0].text(-0.45, 0.5, cond['name'], transform=axes[row_idx, 0].transAxes,
                              fontsize=9, fontweight="bold", va='center', ha='right')

    plt.tight_layout(rect=[0.08, 0.02, 0.98, 0.95])
    output_path = os.path.expanduser("~/ros2_ws/evaluation_results/master_4x4_numbered_matrix.png")
    plt.savefig(output_path)
    print(f"✅ Genummerde 4x4 Master Matrix opgeslagen op: {output_path}")

if __name__ == "__main__":
    generate_4x4_master_numbered()