"""
Powered Evaluation Suite for IPD Recovery Pipeline
Runs each test configuration across multiple random seeds to produce
statistically robust benchmarks with confidence intervals.

Metrics reported:
- IAE (Integrated Absolute Error) with 95% bootstrap CIs
- Curve Fidelity (1 - IAE) as percentage
- RMSE, Median Survival Error, Event Count Error
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from tqdm import tqdm

from pipeline.core import IPDRecoveryPipeline, RiskTable
from pipeline.synthetic import SyntheticTestSuite, SyntheticConfig
from pipeline.evaluation import IPDEvaluator


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000,
                 ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if len(data) == 0:
        return (float('nan'), float('nan'))
    boot_means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    return (float(np.percentile(boot_means, alpha * 100)),
            float(np.percentile(boot_means, (1 - alpha) * 100)))


def run_powered_evaluation(n_seeds: int = 50,
                           output_dir: str = './output/powered_evaluation'):
    """
    Run powered evaluation across multiple random seeds.

    Args:
        n_seeds: Number of random seeds per test configuration
        output_dir: Output directory for results
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("POWERED EVALUATION - IPD Recovery Pipeline")
    print(f"Seeds per configuration: {n_seeds}")
    print("=" * 80)

    # Define test configurations (same as test battery but without generating all at once)
    test_configs = [
        ('basic_single_arm', SyntheticConfig(
            n_patients=100, median_survival=24.0, censoring_rate=0.3, resolution=(800, 600))),
        ('low_resolution', SyntheticConfig(
            n_patients=100, median_survival=24.0, censoring_rate=0.3, resolution=(400, 300), dpi=50)),
        ('high_resolution', SyntheticConfig(
            n_patients=100, median_survival=24.0, censoring_rate=0.3, resolution=(1600, 1200), dpi=150)),
        ('overlapping_curves', SyntheticConfig(
            n_patients=150, median_survival=24.0, censoring_rate=0.25, n_groups=3,
            hazard_ratio=0.85, resolution=(800, 600))),
        ('truncated_y_axis', SyntheticConfig(
            n_patients=100, median_survival=24.0, censoring_rate=0.3,
            truncate_y_axis=True, y_axis_min=0.2, resolution=(800, 600))),
        ('heavy_censoring', SyntheticConfig(
            n_patients=100, median_survival=24.0, censoring_rate=0.7, resolution=(800, 600))),
        ('light_censoring', SyntheticConfig(
            n_patients=100, median_survival=24.0, censoring_rate=0.05, resolution=(800, 600))),
        ('small_sample', SyntheticConfig(
            n_patients=30, median_survival=24.0, censoring_rate=0.3, resolution=(800, 600))),
        ('large_sample', SyntheticConfig(
            n_patients=500, median_survival=24.0, censoring_rate=0.3, resolution=(800, 600))),
        ('short_followup', SyntheticConfig(
            n_patients=100, median_survival=6.0, censoring_rate=0.3, follow_up_time=12.0,
            resolution=(800, 600))),
        ('long_followup', SyntheticConfig(
            n_patients=100, median_survival=48.0, censoring_rate=0.3, follow_up_time=120.0,
            resolution=(800, 600))),
        ('noisy_image', SyntheticConfig(
            n_patients=100, median_survival=24.0, censoring_rate=0.3, add_noise=True,
            noise_level=0.03, resolution=(800, 600))),
        ('two_arm_trial', SyntheticConfig(
            n_patients=120, median_survival=20.0, censoring_rate=0.3, n_groups=2,
            hazard_ratio=0.7, resolution=(800, 600))),
        ('no_risk_table', SyntheticConfig(
            n_patients=100, median_survival=24.0, censoring_rate=0.3,
            include_risk_table=False, resolution=(800, 600))),
    ]

    evaluator = IPDEvaluator(output_dir=output_dir)
    all_results = {}  # config_name -> list of IAEs across seeds

    total_runs = len(test_configs) * n_seeds
    print(f"\nTotal evaluation runs: {total_runs}")
    print()

    with tqdm(total=total_runs, desc="Powered evaluation") as pbar:
        for config_name, config in test_configs:
            seed_iaes = []
            seed_rmses = []
            seed_fidelities = []
            seed_event_errors = []
            seed_censor_errors = []

            for seed in range(n_seeds):
                # Generate test data with this seed
                test_results = _generate_single_case(config_name, config, seed, output_dir)

                if test_results is None:
                    pbar.update(1)
                    continue

                # Run pipeline
                y_min = config.y_axis_min if config.truncate_y_axis else 0.0
                pipeline = IPDRecoveryPipeline(
                    n_curves=config.n_groups,
                    use_preprocessing=True,
                    use_ocr=False
                )

                eval_result = evaluator.evaluate_single_case(test_results, pipeline)

                if eval_result['success'] and eval_result.get('aggregate_metrics', {}).get('success'):
                    iae = eval_result['aggregate_metrics']['mean_iae']
                    rmse = eval_result['aggregate_metrics']['mean_rmse']
                    seed_iaes.append(iae)
                    seed_rmses.append(rmse)
                    seed_fidelities.append(1.0 - iae)
                    seed_event_errors.append(eval_result['aggregate_metrics']['mean_n_events_error'])
                    seed_censor_errors.append(eval_result['aggregate_metrics']['mean_censoring_rate_error'])

                pbar.update(1)

            all_results[config_name] = {
                'iaes': seed_iaes,
                'rmses': seed_rmses,
                'fidelities': seed_fidelities,
                'event_errors': seed_event_errors,
                'censor_errors': seed_censor_errors,
                'n_seeds': len(seed_iaes),
                'n_groups': config.n_groups,
            }

    # Compute aggregate statistics
    _generate_powered_report(all_results, n_seeds, output_dir)
    _generate_powered_plots(all_results, output_dir)

    print(f"\nResults saved to {output_dir}")


def _generate_single_case(config_name: str, config: SyntheticConfig,
                          seed: int, output_dir: str) -> Dict:
    """Generate a single test case with a specific seed."""
    from pipeline.synthetic import SyntheticKMGenerator

    generator = SyntheticKMGenerator(config, seed=seed)
    save_path = os.path.join(output_dir, f"_tmp_{config_name}.png")

    try:
        result = generator.plot_km_curve(save_path)
        result['name'] = config_name
        result['config'] = config
        return result
    except Exception as e:
        return None


def _generate_powered_report(all_results: Dict, n_seeds: int, output_dir: str):
    """Generate comprehensive powered evaluation report."""
    report = []
    report.append("=" * 90)
    report.append("POWERED EVALUATION REPORT - IPD Recovery Pipeline")
    report.append("=" * 90)
    report.append(f"Seeds per configuration: {n_seeds}")
    report.append("")

    # Collect all IAEs across all configs
    all_iaes = []
    all_fidelities = []
    for name, data in all_results.items():
        all_iaes.extend(data['iaes'])
        all_fidelities.extend(data['fidelities'])

    all_iaes = np.array(all_iaes)
    all_fidelities = np.array(all_fidelities)

    if len(all_iaes) == 0:
        report.append("No successful runs.")
        return

    # Overall statistics
    report.append("OVERALL STATISTICS (all configurations pooled)")
    report.append("-" * 90)
    report.append(f"  Total successful runs: {len(all_iaes)}")
    report.append("")

    iae_ci = bootstrap_ci(all_iaes)
    fid_ci = bootstrap_ci(all_fidelities)

    report.append(f"  IAE (Integrated Absolute Error):")
    report.append(f"    Mean:   {np.mean(all_iaes):.4f}  (95% CI: {iae_ci[0]:.4f} - {iae_ci[1]:.4f})")
    report.append(f"    Median: {np.median(all_iaes):.4f}")
    report.append(f"    Std:    {np.std(all_iaes):.4f}")
    report.append(f"    95th percentile: {np.percentile(all_iaes, 95):.4f}")
    report.append("")
    report.append(f"  Curve Fidelity (1 - IAE):")
    report.append(f"    Mean:   {np.mean(all_fidelities)*100:.1f}%  (95% CI: {fid_ci[0]*100:.1f}% - {fid_ci[1]*100:.1f}%)")
    report.append(f"    Median: {np.median(all_fidelities)*100:.1f}%")
    report.append(f"    5th percentile: {np.percentile(all_fidelities, 5)*100:.1f}%")
    report.append("")

    # Comparison to KM-GPT
    report.append("  KM-GPT Benchmarks (arxiv:2509.18141v1):")
    report.append("    Median IAE: 0.018 (95% CI: 0.002 - 0.088)")
    report.append(f"    Our median IAE: {np.median(all_iaes):.4f}")
    pct_within = np.mean(all_iaes <= 0.088) * 100
    report.append(f"    Runs within KM-GPT 95% CI: {pct_within:.1f}%")
    report.append("")

    # Per-configuration results
    report.append("")
    report.append("PER-CONFIGURATION RESULTS")
    report.append("-" * 90)
    report.append(f"{'Configuration':<25s} {'N':>4s} {'Mean IAE':>10s} {'95% CI':>22s} "
                  f"{'Fidelity':>10s} {'Median IAE':>11s}")
    report.append("-" * 90)

    for name, data in sorted(all_results.items(), key=lambda x: np.mean(x[1]['iaes']) if x[1]['iaes'] else 999):
        if not data['iaes']:
            report.append(f"{name:<25s} {'0':>4s} {'FAILED':>10s}")
            continue

        iaes = np.array(data['iaes'])
        ci = bootstrap_ci(iaes)
        fid = np.mean(1.0 - iaes) * 100

        report.append(f"{name:<25s} {len(iaes):>4d} {np.mean(iaes):>10.4f} "
                      f"({ci[0]:.4f} - {ci[1]:.4f}) "
                      f"{fid:>9.1f}% {np.median(iaes):>11.4f}")

    report.append("-" * 90)
    report.append("")

    # Success rate
    total_attempted = sum(d['n_seeds'] for d in all_results.values())  # not right, but close
    report.append(f"Overall success rate: {len(all_iaes)}/{n_seeds * len(all_results)} "
                  f"({len(all_iaes) / (n_seeds * len(all_results)) * 100:.1f}%)")

    # Write report
    report_path = os.path.join(output_dir, 'powered_evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"\nPowered report saved to {report_path}")

    # Also save JSON
    json_results = {
        'n_seeds': n_seeds,
        'n_configurations': len(all_results),
        'total_runs': len(all_iaes),
        'overall': {
            'mean_iae': float(np.mean(all_iaes)),
            'median_iae': float(np.median(all_iaes)),
            'std_iae': float(np.std(all_iaes)),
            'iae_95ci': list(iae_ci),
            'pct95_iae': float(np.percentile(all_iaes, 95)),
            'mean_curve_fidelity': float(np.mean(all_fidelities)),
            'median_curve_fidelity': float(np.median(all_fidelities)),
            'curve_fidelity_95ci': list(fid_ci),
            'pct_within_kmgpt_ci': float(pct_within),
        },
        'per_config': {}
    }

    for name, data in all_results.items():
        if not data['iaes']:
            continue
        iaes = np.array(data['iaes'])
        ci = bootstrap_ci(iaes)
        json_results['per_config'][name] = {
            'n_runs': len(iaes),
            'mean_iae': float(np.mean(iaes)),
            'median_iae': float(np.median(iaes)),
            'std_iae': float(np.std(iaes)),
            'iae_95ci': list(ci),
            'mean_curve_fidelity': float(np.mean(1.0 - iaes)),
            'mean_event_error': float(np.mean(data['event_errors'])),
            'mean_censor_rate_error': float(np.mean(data['censor_errors'])),
        }

    json_path = os.path.join(output_dir, 'powered_evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON results saved to {json_path}")

    # Print summary to console
    print("\n" + "\n".join(report[:35]))


def _generate_powered_plots(all_results: Dict, output_dir: str):
    """Generate visualization plots for powered evaluation."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Collect data
    config_names = []
    config_means = []
    config_cis_low = []
    config_cis_high = []
    all_iaes_flat = []

    for name in sorted(all_results.keys()):
        data = all_results[name]
        if not data['iaes']:
            continue
        iaes = np.array(data['iaes'])
        ci = bootstrap_ci(iaes)
        config_names.append(name)
        config_means.append(np.mean(iaes))
        config_cis_low.append(ci[0])
        config_cis_high.append(ci[1])
        all_iaes_flat.extend(iaes)

    all_iaes_flat = np.array(all_iaes_flat)

    # Plot 1: IAE distribution across all runs
    axes[0, 0].hist(all_iaes_flat, bins=40, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0, 0].axvline(np.median(all_iaes_flat), color='red', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(all_iaes_flat):.4f}')
    axes[0, 0].axvline(0.018, color='green', linestyle=':', linewidth=2,
                       label='KM-GPT median: 0.018')
    axes[0, 0].axvline(0.088, color='orange', linestyle=':', linewidth=2,
                       label='KM-GPT 95th: 0.088')
    axes[0, 0].set_xlabel('IAE')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'IAE Distribution (N={len(all_iaes_flat)} runs)')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Mean IAE per config with 95% CIs
    y_pos = np.arange(len(config_names))
    errors_low = [m - l for m, l in zip(config_means, config_cis_low)]
    errors_high = [h - m for m, h in zip(config_means, config_cis_high)]

    axes[0, 1].barh(y_pos, config_means, xerr=[errors_low, errors_high],
                    alpha=0.7, color='steelblue', capsize=3)
    axes[0, 1].axvline(0.018, color='green', linestyle=':', linewidth=2, label='KM-GPT median')
    axes[0, 1].axvline(0.088, color='orange', linestyle=':', linewidth=2, label='KM-GPT 95th')
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels(config_names, fontsize=8)
    axes[0, 1].set_xlabel('Mean IAE (with 95% CI)')
    axes[0, 1].set_title('IAE by Configuration')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # Plot 3: Curve Fidelity distribution
    fidelities = (1.0 - all_iaes_flat) * 100
    axes[1, 0].hist(fidelities, bins=40, alpha=0.7, edgecolor='black', color='green')
    axes[1, 0].axvline(np.median(fidelities), color='red', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(fidelities):.1f}%')
    axes[1, 0].axvline(95, color='orange', linestyle=':', linewidth=2,
                       label='95% target')
    axes[1, 0].set_xlabel('Curve Fidelity (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Curve Fidelity Distribution (1 - IAE)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Box plot per configuration
    box_data = []
    box_labels = []
    for name in sorted(all_results.keys()):
        data = all_results[name]
        if data['iaes']:
            box_data.append(data['iaes'])
            box_labels.append(name)

    bp = axes[1, 1].boxplot(box_data, vert=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    axes[1, 1].axhline(0.018, color='green', linestyle=':', linewidth=2, label='KM-GPT median')
    axes[1, 1].axhline(0.088, color='orange', linestyle=':', linewidth=2, label='KM-GPT 95th')
    axes[1, 1].set_xticklabels(box_labels, rotation=45, ha='right', fontsize=7)
    axes[1, 1].set_ylabel('IAE')
    axes[1, 1].set_title('IAE Distribution by Configuration')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'powered_evaluation_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {plot_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run powered evaluation')
    parser.add_argument('--seeds', type=int, default=50, help='Seeds per config (default: 50)')
    parser.add_argument('--output', default='./output/powered_evaluation', help='Output directory')
    args = parser.parse_args()

    run_powered_evaluation(n_seeds=args.seeds, output_dir=args.output)
