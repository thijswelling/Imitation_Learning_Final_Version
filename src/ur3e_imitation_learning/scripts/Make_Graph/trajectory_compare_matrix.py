#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_folder(folder_path):
    # Zoek naar alle data.csv bestanden (zowel in raw_demonstrations als evaluation_results)
    files = glob.glob(os.path.join(folder_path, "**", "*data.csv"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(folder_path, "*data.csv"))
    
    if not files:
        return None, []

    peaks = []
    plateaus = []

    for f in files:
        df = pd.read_csv(f)
        if 'fz' not in df.columns and 'f_norm' not in df.columns:
            continue
        
        # Gebruik fz indien aanwezig, anders f_norm
        col = 'fz' if 'fz' in df.columns else 'f_norm'
        fz = np.abs(df[col].to_numpy())
        
        if len(fz) == 0:
            continue

        peaks.append(float(np.max(fz)))

        # Neem het laatste deel van de run als het vasthoudplateau (laatste 30%)
        plateau = fz[int(len(fz) * 0.7):]
        if len(plateau) > 0:
            plateaus.append(float(np.mean(plateau)))

    if not peaks:
        return None, []

    stats = {
        "count": len(peaks),
        "peak_mean": np.mean(peaks),
        "peak_std": np.std(peaks) if len(peaks) > 1 else 0.0,
        "plateau_mean": np.mean(plateaus) if plateaus else 0.0,
        "plateau_std": np.std(plateaus) if len(plateaus) > 1 else 0.0
    }
    return stats, peaks

def main():
    use_case = "trajectory"
    demo_mode = "no_haptic_wave" 
    
    demo_dir = os.path.expanduser(f"~/ros2_ws/raw_demonstrations/{use_case}/{demo_mode}")
    eval_dir = os.path.expanduser(f"~/ros2_ws/evaluation_results/trajectory_golf_no_haptic_no_force")

    print(f"\n" + "=" * 65)
    print(f"  VERGELIJKING METRIEKEN: Demonstraties vs. Imitation Model")
    print(f"=" * 65)

    demo_stats, demo_peaks = analyze_folder(demo_dir)
    eval_stats, eval_peaks = analyze_folder(eval_dir)

    if demo_stats:
        print(f"\n📁 DEMONSTRATIES ({demo_mode.upper()}):")
        print(f"  • Aantal geanalyseerde episodes : {demo_stats['count']}")
        print(f"  • Gemiddelde Piekkracht |Fz|    : {demo_stats['peak_mean']:.2f} ± {demo_stats['peak_std']:.2f} N")
        print(f"  • Vasthoudplateau |Fz| (Gem±SD)   : {demo_stats['plateau_mean']:.2f} ± {demo_stats['plateau_std']:.2f} N")

    if eval_stats:
        print(f"\n🤖 IMITATION LEARNING MODEL (EVALUATIE):")
        print(f"  • Aantal evaluatie-runs         : {eval_stats['count']}")
        print(f"  • Gemiddelde Piekkracht |Fz|    : {eval_stats['peak_mean']:.2f} ± {eval_stats['peak_std']:.2f} N")
        print(f"  • Vasthoudplateau |Fz| (Gem±SD)   : {eval_stats['plateau_mean']:.2f} ± {eval_stats['plateau_std']:.2f} N")

    if demo_stats and eval_stats:
        diff_peak = abs(demo_stats['peak_mean'] - eval_stats['peak_mean'])
        diff_plateau = abs(demo_stats['plateau_mean'] - eval_stats['plateau_mean'])
        print(f"\n📈 AFWIKING TUSSEN MODEL EN DEMO:")
        print(f"  • Verschil in Piekkracht        : {diff_peak:.2f} N")
        print(f"  • Verschil in Vasthoudplateau   : {diff_plateau:.2f} N")

        # Genereer direct een mooie vergelijkingsgrafiek voor je presentatie/scriptie!
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        categories = ['Demonstraties\n(Human)', 'Imitation Model\n(AI Policy)']
        peak_means = [demo_stats['peak_mean'], eval_stats['peak_mean']]
        peak_stds = [demo_stats['peak_std'], eval_stats['peak_std']]
        plateau_means = [demo_stats['plateau_mean'], eval_stats['plateau_mean']]
        plateau_stds = [demo_stats['plateau_std'], eval_stats['plateau_std']]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width/2, peak_means, width, yerr=peak_stds, capsize=5, label='Piekkracht (Peak Fz)', color='#1f77b4', alpha=0.85, edgecolor='black')
        ax.bar(x + width/2, plateau_means, width, yerr=plateau_stds, capsize=5, label='Vasthoudplateau (Plateau Fz)', color='#2ca02c', alpha=0.85, edgecolor='black')

        ax.set_ylabel('Kracht |Fz| (N)')
        ax.set_title('trajectory no haptic no force Vergelijking Krachtmetrieken: Human Demo vs. Imitation Policy')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)

        plt.tight_layout()
        plot_output = os.path.join(eval_dir, "model_vs_demo_comparison.png")
        plt.savefig(plot_output)
        plt.close()
        print(f"\n📊 Grafiek succesvol opgeslagen als afbeelding in: {plot_output}")

    print("=" * 65 + "\n")

if __name__ == '__main__':
    main()