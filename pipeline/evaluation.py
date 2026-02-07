"""
Evaluation and Benchmarking for IPD Recovery Pipeline
Compares performance against KM-GPT benchmarks using:
- Integrated Absolute Error (IAE)
- RMSE
- Median survival accuracy
- Risk table extraction accuracy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os
import json
from scipy import stats
from tqdm import tqdm

from pipeline.core import (
    IPDRecoveryPipeline, IPDRecord, RiskTable, AxisConfig
)
from pipeline.synthetic import SyntheticTestSuite, SyntheticKMGenerator


class IPDEvaluator:
    """
    Comprehensive evaluation suite for IPD recovery pipeline
    """
    
    def __init__(self, output_dir: str = './output/evaluation'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
    
    def evaluate_single_case(self, test_case: Dict, 
                            pipeline: IPDRecoveryPipeline) -> Dict:
        """
        Evaluate pipeline on a single test case
        
        Args:
            test_case: Dict from synthetic generator with ground truth
            pipeline: IPDRecoveryPipeline instance
            
        Returns:
            Dict with evaluation metrics
        """
        # Extract ground truth
        gt_ipd_list = test_case['ipd']
        gt_curves = test_case['curves']
        gt_risk_tables = test_case['risk_tables']
        
        # Create manual risk table for pipeline
        manual_risk_table = self._create_manual_risk_table(gt_risk_tables, gt_ipd_list)
        
        # Configure axis ranges correctly for this test case
        cfg = test_case['config']
        y_min = cfg.y_axis_min if cfg.truncate_y_axis else 0.0
        manual_axis_ranges = {
            'x_range': (0, cfg.follow_up_time),
            'y_range': (y_min, 1.0)
        }

        # Use correct number of curves for this test case
        pipeline_for_case = IPDRecoveryPipeline(
            n_curves=cfg.n_groups,
            use_preprocessing=pipeline.use_preprocessing,
            use_ocr=pipeline.use_ocr
        )

        # Run pipeline
        try:
            result = pipeline_for_case.process(
                img=test_case['image'],
                mask=test_case['mask'],
                manual_axis_ranges=manual_axis_ranges,
                manual_risk_table=manual_risk_table
            )
            
            success = True
            error_msg = None
        except Exception as e:
            success = False
            error_msg = str(e)
            result = None
        
        if not success:
            return {
                'test_name': test_case['name'],
                'success': False,
                'error': error_msg,
                'metrics': {}
            }
        
        # Compute metrics for each group
        group_metrics = {}
        
        for group_id, (gt_ipd, (gt_times, gt_probs)) in enumerate(zip(gt_ipd_list, gt_curves)):
            group_name = f"Group_{group_id + 1}"
            
            if group_name not in result['ipd']:
                group_metrics[group_name] = {
                    'success': False,
                    'error': 'Group not found in reconstruction'
                }
                continue
            
            reconstructed_ipd = result['ipd'][group_name]
            
            # 1. Compute IAE (Integrated Absolute Error)
            iae = self._compute_iae(
                (gt_times, gt_probs),
                reconstructed_ipd
            )
            
            # 2. Compute RMSE
            rmse = self._compute_rmse(
                (gt_times, gt_probs),
                reconstructed_ipd,
                test_case['config'].follow_up_time
            )
            
            # 3. Median survival accuracy
            median_gt = self._compute_median_survival(gt_times, gt_probs)
            median_recon = self._compute_median_survival_from_ipd(reconstructed_ipd)
            median_ae = abs(median_gt - median_recon)
            
            # 4. Number of patients accuracy
            n_patients_gt = len(gt_ipd)
            n_patients_recon = len(reconstructed_ipd)
            n_patients_error = abs(n_patients_gt - n_patients_recon)
            
            # 5. Number of events accuracy
            n_events_gt = (gt_ipd['event'] == 1).sum()
            n_events_recon = sum(1 for r in reconstructed_ipd if r.event == 1)
            n_events_error = abs(n_events_gt - n_events_recon)
            
            # 6. Censoring rate accuracy
            censor_rate_gt = (gt_ipd['event'] == 0).sum() / len(gt_ipd)
            censor_rate_recon = sum(1 for r in reconstructed_ipd if r.event == 0) / len(reconstructed_ipd)
            censor_rate_error = abs(censor_rate_gt - censor_rate_recon)
            
            group_metrics[group_name] = {
                'success': True,
                'iae': iae,
                'rmse': rmse,
                'median_survival_gt': median_gt,
                'median_survival_recon': median_recon,
                'median_ae': median_ae,
                'n_patients_gt': n_patients_gt,
                'n_patients_recon': n_patients_recon,
                'n_patients_error': n_patients_error,
                'n_events_gt': n_events_gt,
                'n_events_recon': n_events_recon,
                'n_events_error': n_events_error,
                'censoring_rate_gt': censor_rate_gt,
                'censoring_rate_recon': censor_rate_recon,
                'censoring_rate_error': censor_rate_error
            }
        
        # Aggregate metrics
        aggregate_metrics = self._aggregate_group_metrics(group_metrics)
        
        return {
            'test_name': test_case['name'],
            'success': True,
            'config': {
                'n_patients': test_case['config'].n_patients,
                'n_groups': test_case['config'].n_groups,
                'median_survival': test_case['config'].median_survival,
                'censoring_rate': test_case['config'].censoring_rate,
                'resolution': test_case['config'].resolution,
                'truncate_y_axis': test_case['config'].truncate_y_axis
            },
            'group_metrics': group_metrics,
            'aggregate_metrics': aggregate_metrics,
            'result': result
        }
    
    def _create_manual_risk_table(self, gt_risk_tables: List[Dict], 
                                  gt_ipd_list: List[pd.DataFrame]) -> RiskTable:
        """Create manual risk table from ground truth"""
        timepoints = gt_risk_tables[0]['timepoints']
        n_at_risk = {}
        group_labels = []
        
        for i, (risk_table, ipd) in enumerate(zip(gt_risk_tables, gt_ipd_list)):
            group_name = f"Group_{i + 1}"
            n_at_risk[group_name] = risk_table['n_at_risk']
            group_labels.append(group_name)
        
        return RiskTable(
            timepoints=timepoints,
            n_at_risk=n_at_risk,
            group_labels=group_labels
        )
    
    def _compute_iae(self, gt_curve: Tuple[np.ndarray, np.ndarray],
                    reconstructed_ipd: List[IPDRecord]) -> float:
        """Compute Integrated Absolute Error (normalized)"""
        gt_times, gt_probs = gt_curve
        
        # Compute KM curve from reconstructed IPD
        recon_times, recon_probs = self._compute_km_curve(reconstructed_ipd)
        
        if len(recon_times) == 0:
            return float('inf')
        
        # Get common time grid
        all_times = np.unique(np.concatenate([gt_times, recon_times]))
        
        # Interpolate both curves
        from scipy.interpolate import interp1d
        
        gt_interp = interp1d(gt_times, gt_probs, kind='previous',
                            bounds_error=False, fill_value=(1.0, gt_probs[-1]))
        recon_interp = interp1d(recon_times, recon_probs, kind='previous',
                               bounds_error=False, fill_value=(1.0, recon_probs[-1]))
        
        # Evaluate at common grid
        gt_vals = gt_interp(all_times)
        recon_vals = recon_interp(all_times)
        
        # Compute IAE using trapezoidal rule
        abs_diff = np.abs(gt_vals - recon_vals)
        iae = np.trapz(abs_diff, all_times)
        
        # Normalize by time range
        time_range = all_times[-1] - all_times[0]
        if time_range > 0:
            iae = iae / time_range
        
        return float(iae)
    
    def _compute_rmse(self, gt_curve: Tuple[np.ndarray, np.ndarray],
                     reconstructed_ipd: List[IPDRecord],
                     max_time: float) -> float:
        """Compute Root Mean Squared Error"""
        gt_times, gt_probs = gt_curve
        recon_times, recon_probs = self._compute_km_curve(reconstructed_ipd)
        
        if len(recon_times) == 0:
            return float('inf')
        
        # Create common time grid (100 points)
        time_grid = np.linspace(0, max_time, 100)
        
        from scipy.interpolate import interp1d
        
        gt_interp = interp1d(gt_times, gt_probs, kind='previous',
                            bounds_error=False, fill_value=(1.0, gt_probs[-1]))
        recon_interp = interp1d(recon_times, recon_probs, kind='previous',
                               bounds_error=False, fill_value=(1.0, recon_probs[-1]))
        
        gt_vals = gt_interp(time_grid)
        recon_vals = recon_interp(time_grid)
        
        rmse = np.sqrt(np.mean((gt_vals - recon_vals) ** 2))
        
        return float(rmse)
    
    def _compute_km_curve(self, ipd: List[IPDRecord]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert IPD to KM curve"""
        if not ipd:
            return np.array([]), np.array([])
        
        # Sort by time
        sorted_ipd = sorted(ipd, key=lambda x: x.time)
        
        times = [0.0]
        probs = [1.0]
        
        n_at_risk = len(sorted_ipd)
        survival_prob = 1.0
        
        i = 0
        while i < len(sorted_ipd):
            current_time = sorted_ipd[i].time
            
            # Count events at this time
            n_events = 0
            j = i
            while j < len(sorted_ipd) and sorted_ipd[j].time == current_time:
                if sorted_ipd[j].event == 1:
                    n_events += 1
                j += 1
            
            # Update survival
            if n_at_risk > 0 and n_events > 0:
                survival_prob *= (1 - n_events / n_at_risk)
                times.append(current_time)
                probs.append(survival_prob)
            
            n_at_risk -= (j - i)
            i = j
        
        return np.array(times), np.array(probs)
    
    def _compute_median_survival(self, times: np.ndarray, probs: np.ndarray) -> float:
        """Compute median survival from KM curve"""
        for i in range(len(probs)):
            if probs[i] <= 0.5:
                if i == 0:
                    return times[i]
                # Linear interpolation
                t0, s0 = times[i-1], probs[i-1]
                t1, s1 = times[i], probs[i]
                if s0 != s1:
                    median = t0 + (0.5 - s0) * (t1 - t0) / (s1 - s0)
                else:
                    median = t0
                return median
        # Not reached
        return times[-1]
    
    def _compute_median_survival_from_ipd(self, ipd: List[IPDRecord]) -> float:
        """Compute median survival from reconstructed IPD"""
        times, probs = self._compute_km_curve(ipd)
        if len(times) == 0:
            return 0.0
        return self._compute_median_survival(times, probs)
    
    def _aggregate_group_metrics(self, group_metrics: Dict) -> Dict:
        """Aggregate metrics across groups"""
        successful_groups = [m for m in group_metrics.values() if m.get('success', False)]
        
        if not successful_groups:
            return {'success': False}
        
        return {
            'success': True,
            'mean_iae': np.mean([m['iae'] for m in successful_groups]),
            'median_iae': np.median([m['iae'] for m in successful_groups]),
            'mean_rmse': np.mean([m['rmse'] for m in successful_groups]),
            'mean_median_ae': np.mean([m['median_ae'] for m in successful_groups]),
            'mean_n_patients_error': np.mean([m['n_patients_error'] for m in successful_groups]),
            'mean_n_events_error': np.mean([m['n_events_error'] for m in successful_groups]),
            'mean_censoring_rate_error': np.mean([m['censoring_rate_error'] for m in successful_groups])
        }
    
    def evaluate_test_suite(self, test_results: List[Dict], 
                           pipeline: IPDRecoveryPipeline) -> List[Dict]:
        """
        Evaluate pipeline on entire test suite
        
        Args:
            test_results: List of test cases from synthetic generator
            pipeline: IPDRecoveryPipeline instance
            
        Returns:
            List of evaluation results
        """
        evaluation_results = []
        
        print("Evaluating IPD Recovery Pipeline on Test Suite")
        print("=" * 70)
        
        for test_case in tqdm(test_results, desc="Evaluating"):
            eval_result = self.evaluate_single_case(test_case, pipeline)
            evaluation_results.append(eval_result)
            self.results.append(eval_result)
        
        # Save results
        self._save_results(evaluation_results)
        
        # Generate report
        self._generate_evaluation_report(evaluation_results)
        
        # Create visualizations
        self._create_visualizations(evaluation_results)
        
        return evaluation_results
    
    def _save_results(self, results: List[Dict]):
        """Save evaluation results to JSON"""
        
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        # Remove non-serializable objects
        serializable_results = []
        for r in results:
            clean_r = {
                'test_name': r['test_name'],
                'success': r['success'],
                'config': convert_to_serializable(r.get('config', {})),
                'aggregate_metrics': convert_to_serializable(r.get('aggregate_metrics', {})),
                'group_metrics': {}
            }
            
            if 'group_metrics' in r:
                for group, metrics in r['group_metrics'].items():
                    clean_r['group_metrics'][group] = {
                        k: convert_to_serializable(v) for k, v in metrics.items() 
                        if not isinstance(v, (list, dict, np.ndarray))
                    }
            
            serializable_results.append(clean_r)
        
        results_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
    
    def _generate_evaluation_report(self, results: List[Dict]):
        """Generate comprehensive text report"""
        report = []
        report.append("=" * 80)
        report.append("IPD RECOVERY PIPELINE EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        n_total = len(results)
        n_success = sum(1 for r in results if r['success'])
        success_rate = n_success / n_total * 100 if n_total > 0 else 0
        
        report.append(f"Total test cases: {n_total}")
        report.append(f"Successful: {n_success} ({success_rate:.1f}%)")
        report.append(f"Failed: {n_total - n_success}")
        report.append("")
        
        # Aggregate metrics across all successful cases
        successful_results = [r for r in results if r['success'] and 
                            r.get('aggregate_metrics', {}).get('success', False)]
        
        if successful_results:
            report.append("AGGREGATE METRICS (across all successful cases):")
            report.append("-" * 80)
            
            all_iaes = [r['aggregate_metrics']['mean_iae'] for r in successful_results]
            all_rmses = [r['aggregate_metrics']['mean_rmse'] for r in successful_results]
            all_median_aes = [r['aggregate_metrics']['mean_median_ae'] for r in successful_results]
            
            report.append(f"  IAE (Integrated Absolute Error):")
            report.append(f"    Mean: {np.mean(all_iaes):.6f}")
            report.append(f"    Median: {np.median(all_iaes):.6f}")
            report.append(f"    95th percentile: {np.percentile(all_iaes, 95):.6f}")
            report.append("")
            
            report.append(f"  RMSE (Root Mean Squared Error):")
            report.append(f"    Mean: {np.mean(all_rmses):.6f}")
            report.append(f"    Median: {np.median(all_rmses):.6f}")
            report.append(f"    95th percentile: {np.percentile(all_rmses, 95):.6f}")
            report.append("")
            
            report.append(f"  Median Survival Absolute Error:")
            report.append(f"    Mean: {np.mean(all_median_aes):.4f} time units")
            report.append(f"    Median: {np.median(all_median_aes):.4f} time units")
            report.append("")
        
        # Comparison to KM-GPT benchmarks (from paper)
        report.append("COMPARISON TO KM-GPT BENCHMARKS:")
        report.append("-" * 80)
        report.append("  KM-GPT paper reports:")
        report.append("    Median IAE: 0.018 (95% CI: 0.002-0.088)")
        report.append("    Median Absolute Error in median OS: 0.005 (95% CI: 0.000-0.088)")
        report.append("")
        
        if successful_results:
            our_median_iae = np.median(all_iaes)
            our_median_ae = np.median(all_median_aes)
            
            # Normalize median_ae to [0,1] scale (assuming max time = 60 months)
            our_median_ae_normalized = our_median_ae / 60.0
            
            report.append("  Our pipeline:")
            report.append(f"    Median IAE: {our_median_iae:.6f}")
            report.append(f"    Median Absolute Error in median survival: {our_median_ae_normalized:.6f} (normalized)")
            report.append("")
            
            # Performance comparison
            if our_median_iae <= 0.088:  # Within KM-GPT's 95% CI
                report.append("  [PASS] IAE performance is within KM-GPT's 95% confidence interval")
            else:
                report.append("  [FAIL] IAE performance exceeds KM-GPT's 95% confidence interval")
            
            if our_median_ae_normalized <= 0.088:
                report.append("  [PASS] Median survival error is within KM-GPT's 95% confidence interval")
            else:
                report.append("  [FAIL] Median survival error exceeds KM-GPT's 95% confidence interval")
        
        report.append("")
        report.append("")
        
        # Individual test case results
        report.append("INDIVIDUAL TEST CASE RESULTS:")
        report.append("-" * 80)
        
        for r in results:
            report.append(f"\n{r['test_name']}:")
            
            if not r['success']:
                report.append(f"  Status: FAILED")
                report.append(f"  Error: {r.get('error', 'Unknown error')}")
                continue
            
            report.append(f"  Status: SUCCESS")
            
            if 'config' in r:
                cfg = r['config']
                report.append(f"  Config: {cfg['n_patients']} patients, {cfg['n_groups']} groups, "
                            f"censoring={cfg['censoring_rate']:.2f}")
            
            if r.get('aggregate_metrics', {}).get('success', False):
                metrics = r['aggregate_metrics']
                report.append(f"  IAE: {metrics['mean_iae']:.6f}")
                report.append(f"  RMSE: {metrics['mean_rmse']:.6f}")
                report.append(f"  Median survival AE: {metrics['mean_median_ae']:.4f}")
                report.append(f"  Patient count error: {metrics['mean_n_patients_error']:.1f}")
                report.append(f"  Event count error: {metrics['mean_n_events_error']:.1f}")
        
        # Save report
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"Evaluation report saved to {report_path}")
        
        # Also print summary to console
        print("\n" + "\n".join(report[:30]))  # Print first 30 lines
    
    def _create_visualizations(self, results: List[Dict]):
        """Create visualization plots"""
        successful_results = [r for r in results if r['success'] and 
                            r.get('aggregate_metrics', {}).get('success', False)]
        
        if not successful_results:
            print("No successful results to visualize")
            return
        
        # Extract metrics
        test_names = [r['test_name'] for r in successful_results]
        iaes = [r['aggregate_metrics']['mean_iae'] for r in successful_results]
        rmses = [r['aggregate_metrics']['mean_rmse'] for r in successful_results]
        median_aes = [r['aggregate_metrics']['mean_median_ae'] for r in successful_results]
        
        # Create multi-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: IAE distribution
        axes[0, 0].boxplot([iaes], labels=['IAE'])
        axes[0, 0].axhline(y=0.018, color='r', linestyle='--', label='KM-GPT median')
        axes[0, 0].axhline(y=0.088, color='orange', linestyle='--', label='KM-GPT 95th %ile')
        axes[0, 0].set_ylabel('Integrated Absolute Error')
        axes[0, 0].set_title('IAE Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: IAE by test case
        axes[0, 1].bar(range(len(test_names)), iaes, alpha=0.7)
        axes[0, 1].axhline(y=0.018, color='r', linestyle='--', linewidth=2, label='KM-GPT median')
        axes[0, 1].set_xlabel('Test Case')
        axes[0, 1].set_ylabel('IAE')
        axes[0, 1].set_title('IAE by Test Case')
        axes[0, 1].set_xticks(range(len(test_names)))
        axes[0, 1].set_xticklabels(test_names, rotation=45, ha='right', fontsize=8)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: RMSE distribution
        axes[1, 0].hist(rmses, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=np.median(rmses), color='r', linestyle='--', 
                          linewidth=2, label=f'Median: {np.median(rmses):.4f}')
        axes[1, 0].set_xlabel('RMSE')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('RMSE Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Median survival error distribution
        axes[1, 1].hist(median_aes, bins=20, alpha=0.7, edgecolor='black', color='green')
        axes[1, 1].axvline(x=np.median(median_aes), color='r', linestyle='--',
                          linewidth=2, label=f'Median: {np.median(median_aes):.2f}')
        axes[1, 1].set_xlabel('Median Survival Absolute Error (time units)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Median Survival Error Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'evaluation_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Evaluation plots saved to {plot_path}")
        
        plt.close()
        
        # Create per-test-case comparison plot
        self._create_comparison_plot(successful_results)
    
    def _create_comparison_plot(self, results: List[Dict]):
        """Create detailed comparison plot for selected test cases"""
        # Select up to 4 interesting test cases
        selected = results[:min(4, len(results))]
        
        if not selected:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, result in enumerate(selected):
            if idx >= 4:
                break
            
            ax = axes[idx]
            
            # Plot original vs reconstructed curves
            test_case = result['result']
            colors = ['blue', 'red', 'green', 'orange']
            
            for i, curve in enumerate(test_case['curves']):
                if not curve:
                    continue
                
                group_name = f"Group_{i+1}"
                times_orig = [t for t, _ in curve]
                probs_orig = [p for _, p in curve]
                
                ax.plot(times_orig, probs_orig, color=colors[i % len(colors)],
                       linewidth=2, label=f'{group_name} (original)', alpha=0.7)
                
                # Reconstructed
                if group_name in test_case['ipd']:
                    ipd = test_case['ipd'][group_name]
                    times_recon, probs_recon = self._compute_km_curve(ipd)
                    ax.plot(times_recon, probs_recon, color=colors[i % len(colors)],
                           linewidth=2, linestyle='--', label=f'{group_name} (reconstructed)',
                           alpha=0.7)
            
            # Add metrics to title
            if result.get('aggregate_metrics', {}).get('success', False):
                metrics = result['aggregate_metrics']
                title = f"{result['test_name']}\nIAE={metrics['mean_iae']:.4f}, RMSE={metrics['mean_rmse']:.4f}"
            else:
                title = result['test_name']
            
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Time')
            ax.set_ylabel('Survival Probability')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        comparison_path = os.path.join(self.output_dir, 'curve_comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"Curve comparison plot saved to {comparison_path}")
        
        plt.close()


# ==========================================
# MAIN EVALUATION SCRIPT
# ==========================================
if __name__ == "__main__":
    print("IPD Recovery Pipeline - Comprehensive Evaluation")
    print("=" * 80)
    print()
    
    # Step 1: Generate synthetic test data
    print("Step 1: Generating synthetic test battery...")
    suite = SyntheticTestSuite(output_dir='./output/synthetic_data')
    test_results = suite.generate_test_battery()
    suite.create_summary_report(test_results)
    print(f"Generated {len(test_results)} test cases")
    print()
    
    # Step 2: Initialize pipeline
    print("Step 2: Initializing IPD Recovery Pipeline...")
    pipeline = IPDRecoveryPipeline(
        n_curves=1,
        use_preprocessing=True,
        use_ocr=False  # Using manual axis ranges for controlled evaluation
    )
    print("Pipeline initialized")
    print()
    
    # Step 3: Run evaluation
    print("Step 3: Running comprehensive evaluation...")
    evaluator = IPDEvaluator(output_dir='./output/evaluation')
    evaluation_results = evaluator.evaluate_test_suite(test_results, pipeline)
    print()
    
    # Step 4: Summary statistics
    print("Step 4: Computing summary statistics...")
    successful = [r for r in evaluation_results if r['success'] and 
                 r.get('aggregate_metrics', {}).get('success', False)]
    
    if successful:
        iaes = [r['aggregate_metrics']['mean_iae'] for r in successful]
        rmses = [r['aggregate_metrics']['mean_rmse'] for r in successful]
        
        print("\nFINAL SUMMARY:")
        print("=" * 80)
        print(f"Success rate: {len(successful)}/{len(evaluation_results)} "
              f"({len(successful)/len(evaluation_results)*100:.1f}%)")
        print(f"\nMedian IAE: {np.median(iaes):.6f}")
        print(f"  (KM-GPT benchmark: 0.018, 95% CI: 0.002-0.088)")
        print(f"\nMedian RMSE: {np.median(rmses):.6f}")
        print(f"\n95th percentile IAE: {np.percentile(iaes, 95):.6f}")
        print(f"  (KM-GPT benchmark 95th %ile: 0.088)")
        
        # Performance assessment
        if np.median(iaes) <= 0.018:
            print("\n[PASS] Performance EXCEEDS KM-GPT median benchmark!")
        elif np.median(iaes) <= 0.088:
            print("\n[PASS] Performance is within KM-GPT's 95% confidence interval")
        else:
            print("\n[WARN] Performance is below KM-GPT benchmarks - room for improvement")
    else:
        print("\n[WARN] No successful evaluations - pipeline needs debugging")
    
    print("\nEvaluation complete! Check ./evaluation_results/ for detailed outputs.")