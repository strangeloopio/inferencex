"""
Plot GPU/User configuration benchmark results
Generates a 5-panel bar chart comparing:
1. Throughput per GPU (tok/s/gpu)
2. End-to-End Latency (s)
3. User Interactivity (tok/s/user)
4. Time to First Token (s)
5. Power Efficiency (tokens/kW) - from real nvidia-smi data
"""

import argparse
import csv
import glob
import json
import os

import matplotlib.pyplot as plt


def load_results(filepath: str = "benchmark_gpu_users_results.json") -> list:
    """Load GPU/user benchmark results from JSON file"""
    with open(filepath, "r") as f:
        return json.load(f)


def load_power_logs(power_log_dir: str = ".") -> dict:
    """
    Load power logs and calculate average power per GPU configuration.
    Returns dict: {n_gpu: avg_power_per_gpu_watts}
    """
    power_data = {}

    # Find all power log CSV files
    log_files = glob.glob(os.path.join(power_log_dir, "power_log_*gpu_*.csv"))

    for log_file in log_files:
        # Extract n_gpu from filename (e.g., power_log_1gpu_20251130_083607.csv)
        basename = os.path.basename(log_file)
        try:
            n_gpu = int(basename.split("_")[2].replace("gpu", ""))
        except (IndexError, ValueError):
            continue

        # Read power data
        power_readings = []
        with open(log_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    power_w = float(row["power_draw_w"].strip())
                    gpu_util = float(row["gpu_util_pct"].strip())
                    # Only include readings with actual GPU activity (util > 5%)
                    if gpu_util > 5:
                        power_readings.append(power_w)
                except (KeyError, ValueError):
                    continue

        if power_readings:
            avg_power = sum(power_readings) / len(power_readings)
            # Keep the highest average if multiple logs exist for same config
            if n_gpu not in power_data or avg_power > power_data[n_gpu]:
                power_data[n_gpu] = avg_power

    return power_data


def plot_bar_comparison(
    results: list,
    model_name: str,
    gpu_type: str,
    power_data: dict = None,
    save_path: str = "gpu_users_bar_comparison.png"
):
    """Create 5-panel bar chart comparing GPU/user configurations"""
    labels = [r["config_label"] for r in results]

    # Extract ISL/OSL from results (assume all configs have same ISL/OSL)
    isl = results[0].get("isl", "N/A") if results else "N/A"
    osl = results[0].get("osl", "N/A") if results else "N/A"
    throughputs_per_gpu = [r["throughput_per_gpu"] for r in results]
    throughputs_total = [r["throughput_total"] for r in results]
    latencies = [r["avg_latency_s"] for r in results]
    interactivities = [r["interactivity_tokens_per_sec_per_user"] for r in results]
    ttfts = [r["avg_ttft_s"] for r in results]
    n_gpus = [r["n_gpu"] for r in results]

    colors_map = {1: 'steelblue', 2: 'seagreen', 4: 'indianred', 8: 'darkorange'}
    bar_colors = [colors_map.get(g, 'gray') for g in n_gpus]

    # Calculate power efficiency if power data available
    has_power_data = power_data and len(power_data) > 0
    if has_power_data:
        # tokens per kW = throughput_total / (n_gpu * power_per_gpu / 1000)
        power_efficiencies = []
        for r in results:
            n_gpu = r["n_gpu"]
            if n_gpu in power_data:
                total_power_kw = (n_gpu * power_data[n_gpu]) / 1000
                efficiency = r["throughput_total"] / total_power_kw
                power_efficiencies.append(efficiency)
            else:
                power_efficiencies.append(0)
        n_panels = 5
    else:
        n_panels = 4

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5))

    # Panel 1: Throughput per GPU
    axes[0].bar(labels, throughputs_per_gpu, color=bar_colors, edgecolor='white', linewidth=1.5)
    axes[0].set_ylabel('Throughput (tokens/s/GPU)', fontsize=11)
    axes[0].set_title('Throughput per GPU', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(throughputs_per_gpu):
        axes[0].text(i, v + max(throughputs_per_gpu) * 0.02, f'{v:.0f}', ha='center', fontsize=9)

    # Panel 2: Latency
    axes[1].bar(labels, latencies, color=bar_colors, edgecolor='white', linewidth=1.5)
    axes[1].set_ylabel('Latency (seconds)', fontsize=11)
    axes[1].set_title('End-to-End Latency', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(latencies):
        axes[1].text(i, v + max(latencies) * 0.02, f'{v:.1f}', ha='center', fontsize=9)

    # Panel 3: Interactivity
    axes[2].bar(labels, interactivities, color=bar_colors, edgecolor='white', linewidth=1.5)
    axes[2].set_ylabel('Interactivity (tokens/s/user)', fontsize=11)
    axes[2].set_title('User Interactivity', fontsize=12, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    for i, v in enumerate(interactivities):
        axes[2].text(i, v + max(interactivities) * 0.02, f'{v:.1f}', ha='center', fontsize=9)

    # Panel 4: Time to First Token
    axes[3].bar(labels, ttfts, color=bar_colors, edgecolor='white', linewidth=1.5)
    axes[3].set_ylabel('TTFT (seconds)', fontsize=11)
    axes[3].set_title('Time to First Token', fontsize=12, fontweight='bold')
    axes[3].tick_params(axis='x', rotation=45)
    for i, v in enumerate(ttfts):
        axes[3].text(i, v + max(ttfts) * 0.02, f'{v:.2f}', ha='center', fontsize=9)

    # Panel 5: Power Efficiency (if power data available)
    if has_power_data:
        axes[4].bar(labels, power_efficiencies, color=bar_colors, edgecolor='white', linewidth=1.5)
        axes[4].set_ylabel('Efficiency (tokens/s/kW)', fontsize=11)
        axes[4].set_title('Power Efficiency', fontsize=12, fontweight='bold')
        axes[4].tick_params(axis='x', rotation=45)
        for i, v in enumerate(power_efficiencies):
            if v > 0:
                axes[4].text(i, v + max(power_efficiencies) * 0.02, f'{v:.0f}', ha='center', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    unique_gpus = sorted(set(n_gpus))
    legend_elements = [Patch(facecolor=colors_map.get(g, 'gray'), label=f'{g} GPU') for g in unique_gpus]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))

    # Title: Model, GPU, ISL/OSL, Power
    title = f'{model_name} on {gpu_type.upper()}'
    title += f'\nISL: {isl}, OSL: {osl}'
    if has_power_data:
        power_str = ", ".join([f"{g}GPU: {power_data[g]:.0f}W" for g in sorted(power_data.keys())])
        title += f' | Measured Power: {power_str}'
    fig.suptitle(title, fontsize=14, y=1.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate GPU/User benchmark comparison chart")
    parser.add_argument("--gpu", type=str, default="H100",
                        help="GPU type for chart title (default: H100)")
    parser.add_argument("--model", type=str, default="Qwen3-8B-FP8",
                        help="Model name for chart title (default: Qwen3-8B-FP8)")
    parser.add_argument("--results", type=str, default="benchmark_gpu_users_results.json",
                        help="Path to benchmark results JSON file")
    parser.add_argument("--power-logs", type=str, default=".",
                        help="Directory containing power log CSV files (default: current dir)")
    parser.add_argument("--output", type=str, default="gpu_users_bar_comparison.png",
                        help="Output PNG file path")

    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found.")
        print("Run benchmark_gpu_users.py first to collect data.")
        return

    print(f"Loading results from {args.results}")
    results = load_results(args.results)

    # Load power logs if available
    print(f"Looking for power logs in {args.power_logs}")
    power_data = load_power_logs(args.power_logs)
    if power_data:
        print(f"Found power data for {len(power_data)} GPU configuration(s):")
        for n_gpu, avg_power in sorted(power_data.items()):
            print(f"  {n_gpu} GPU: {avg_power:.1f}W avg per GPU")
    else:
        print("No power logs found, skipping power efficiency panel")

    # Print data summary
    print(f"\nConfiguration: {args.model} on {args.gpu.upper()}")
    print("\nData points:")
    print(f"{'Config':<12} {'GPUs':<6} {'Users':<8} {'Thru/GPU':<14} {'Latency':<12} {'Interactivity':<14} {'TTFT':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['config_label']:<12} {r['n_gpu']:<6} {r['n_users']:<8} {r['throughput_per_gpu']:<14.2f} {r['avg_latency_s']:<12.2f} {r['interactivity_tokens_per_sec_per_user']:<14.2f} {r['avg_ttft_s']:<10.2f}")
    print()

    # Generate the bar chart
    plot_bar_comparison(
        results=results,
        model_name=args.model,
        gpu_type=args.gpu,
        power_data=power_data,
        save_path=args.output
    )

    print("Plot generated successfully!")


if __name__ == "__main__":
    main()
