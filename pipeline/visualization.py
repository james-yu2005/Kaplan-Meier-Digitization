"""
Visualization Tool for IPD Recovery Pipeline
Overlays reconstructed KM curves onto original images for visual validation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import cv2
from typing import List, Dict, Tuple, Optional
import os
from pipeline.core import IPDRecord, AxisConfig


class ReconstructionVisualizer:
    """
    Visualize IPD reconstruction results by overlaying curves on original images
    """
    
    def __init__(self, output_dir: str = './visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_reconstruction(self, 
                                img: np.ndarray,
                                original_curves: List[List[Tuple[float, float]]],
                                reconstructed_ipd: Dict[str, List[IPDRecord]],
                                axis_config: AxisConfig,
                                test_name: str = "reconstruction",
                                metrics: Optional[Dict] = None):
        """
        Create comprehensive visualization of reconstruction results
        
        Args:
            img: Original KM plot image
            original_curves: List of original (time, prob) curves
            reconstructed_ipd: Dict mapping group names to IPD records
            axis_config: Axis configuration for coordinate transformation
            test_name: Name for saving the visualization
            metrics: Optional dict with IAE, RMSE, etc.
        """
        # Create figure with multiple panels
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Original image with extracted curves
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_image_with_curves(ax1, img, original_curves, axis_config, 
                                     title="Original Image + Extracted Curves")
        
        # Panel 2: Survival curves comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_survival_comparison(ax2, original_curves, reconstructed_ipd, 
                                       title="Survival Curves: Original vs Reconstructed")
        
        # Panel 3: Difference plot
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_absolute_difference(ax3, original_curves, reconstructed_ipd,
                                       title="Absolute Difference Over Time")
        
        # Panel 4: Image with reconstructed overlay
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_image_with_reconstructed(ax4, img, reconstructed_ipd, 
                                           axis_config, title="Reconstructed Curves Overlay")
        
        # Panel 5: Number at risk verification
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_risk_set_sizes(ax5, reconstructed_ipd, 
                                  title="Reconstructed Risk Set Sizes")
        
        # Panel 6: Metrics summary
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_metrics_summary(ax6, metrics, original_curves, reconstructed_ipd)
        
        # Overall title
        fig.suptitle(f'IPD Reconstruction: {test_name}', fontsize=16, fontweight='bold')
        
        # Save
        save_path = os.path.join(self.output_dir, f'{test_name}_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Visualization saved to {save_path}")
        
        plt.close(fig)
        
        return save_path
    
    def _plot_image_with_curves(self, ax, img, curves, axis_config, title):
        """Plot original image with extracted curves overlaid"""
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        
        colors = ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080']
        
        for i, curve in enumerate(curves):
            if not curve:
                continue
            
            # Convert to pixel coordinates
            xs, ys = [], []
            for time, prob in curve:
                x_px, y_px = self._real_to_pixel(time, prob, axis_config)
                xs.append(x_px)
                ys.append(y_px)
            
            if xs:
                ax.plot(xs, ys, color=colors[i % len(colors)], 
                       linewidth=3, label=f'Curve {i+1}', alpha=0.8)
        
        ax.legend(loc='upper right', fontsize=9)
    
    def _plot_image_with_reconstructed(self, ax, img, reconstructed_ipd, 
                                       axis_config, title):
        """Plot original image with reconstructed curves overlaid"""
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        
        colors = ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080']
        
        for i, (group_name, ipd) in enumerate(reconstructed_ipd.items()):
            # Compute KM curve from IPD
            times, probs = self._compute_km_curve(ipd)
            
            if len(times) == 0:
                continue
            
            # Convert to pixel coordinates
            xs, ys = [], []
            for time, prob in zip(times, probs):
                x_px, y_px = self._real_to_pixel(time, prob, axis_config)
                xs.append(x_px)
                ys.append(y_px)
            
            if xs:
                ax.plot(xs, ys, color=colors[i % len(colors)], 
                       linewidth=3, linestyle='--', label=group_name, alpha=0.8)
        
        ax.legend(loc='upper right', fontsize=9)
    
    def _plot_survival_comparison(self, ax, original_curves, reconstructed_ipd, title):
        """Plot survival curves: original vs reconstructed"""
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Survival Probability', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Plot original curves
        for i, curve in enumerate(original_curves):
            if not curve:
                continue
            
            times = [t for t, _ in curve]
            probs = [p for _, p in curve]
            
            ax.plot(times, probs, color=colors[i % len(colors)],
                   linewidth=2.5, label=f'Original {i+1}', alpha=0.7)
        
        # Plot reconstructed curves
        for i, (group_name, ipd) in enumerate(reconstructed_ipd.items()):
            times, probs = self._compute_km_curve(ipd)
            
            if len(times) > 0:
                ax.plot(times, probs, color=colors[i % len(colors)],
                       linewidth=2.5, linestyle='--', label=f'Recon {i+1}', alpha=0.7)
        
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best', fontsize=9)
    
    def _plot_absolute_difference(self, ax, original_curves, reconstructed_ipd, title):
        """Plot absolute difference between curves over time"""
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('|Original - Reconstructed|', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        from scipy.interpolate import interp1d
        
        for i, curve in enumerate(original_curves):
            if not curve or i >= len(reconstructed_ipd):
                continue
            
            group_name = list(reconstructed_ipd.keys())[i]
            ipd = reconstructed_ipd[group_name]
            
            # Original curve
            orig_times = np.array([t for t, _ in curve])
            orig_probs = np.array([p for _, p in curve])
            
            # Reconstructed curve
            recon_times, recon_probs = self._compute_km_curve(ipd)
            
            if len(recon_times) == 0:
                continue
            
            # Common time grid
            min_time = min(orig_times[0], recon_times[0])
            max_time = max(orig_times[-1], recon_times[-1])
            time_grid = np.linspace(min_time, max_time, 200)
            
            # Interpolate
            orig_interp = interp1d(orig_times, orig_probs, kind='previous',
                                  bounds_error=False, fill_value=(1.0, orig_probs[-1]))
            recon_interp = interp1d(recon_times, recon_probs, kind='previous',
                                   bounds_error=False, fill_value=(1.0, recon_probs[-1]))
            
            orig_vals = orig_interp(time_grid)
            recon_vals = recon_interp(time_grid)
            
            abs_diff = np.abs(orig_vals - recon_vals)
            
            ax.fill_between(time_grid, 0, abs_diff, color=colors[i % len(colors)],
                           alpha=0.3, label=f'Curve {i+1}')
            ax.plot(time_grid, abs_diff, color=colors[i % len(colors)],
                   linewidth=2, alpha=0.8)
        
        ax.legend(loc='best', fontsize=9)
    
    def _plot_risk_set_sizes(self, ax, reconstructed_ipd, title):
        """Plot number at risk over time from reconstructed IPD"""
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Number at Risk', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (group_name, ipd) in enumerate(reconstructed_ipd.items()):
            if not ipd:
                continue
            
            # Sort by time
            sorted_ipd = sorted(ipd, key=lambda x: x.time)
            
            # Compute risk set sizes over time
            times = [0.0]
            n_at_risk = [len(ipd)]
            
            for j, record in enumerate(sorted_ipd):
                times.append(record.time)
                n_at_risk.append(len(ipd) - j - 1)
            
            ax.step(times, n_at_risk, where='post', color=colors[i % len(colors)],
                   linewidth=2, label=group_name, alpha=0.8)
        
        ax.legend(loc='best', fontsize=9)
        ax.set_ylim(bottom=0)
    
    def _plot_metrics_summary(self, ax, metrics, original_curves, reconstructed_ipd):
        """Display metrics summary as text"""
        ax.axis('off')
        
        if metrics is None or not metrics.get('success', False):
            ax.text(0.5, 0.5, 'Metrics not available', 
                   ha='center', va='center', fontsize=12)
            return
        
        # Compute additional statistics
        text_lines = []
        text_lines.append("RECONSTRUCTION METRICS")
        text_lines.append("=" * 40)
        text_lines.append("")
        
        # Core metrics
        if 'mean_iae' in metrics:
            text_lines.append(f"IAE: {metrics['mean_iae']:.6f}")
            
            # Compare to KM-GPT benchmark
            if metrics['mean_iae'] <= 0.018:
                status = "[PASS] Excellent (< KM-GPT median)"
            elif metrics['mean_iae'] <= 0.088:
                status = "[PASS] Good (within KM-GPT 95% CI)"
            else:
                status = "[WARN] Needs improvement"
            text_lines.append(f"  {status}")
            text_lines.append("")
        
        if 'mean_rmse' in metrics:
            text_lines.append(f"RMSE: {metrics['mean_rmse']:.6f}")
            text_lines.append("")
        
        if 'mean_median_ae' in metrics:
            text_lines.append(f"Median Survival AE: {metrics['mean_median_ae']:.4f}")
            text_lines.append("")
        
        # Patient counts
        if 'mean_n_patients_error' in metrics:
            text_lines.append(f"Patient Count Error: {metrics['mean_n_patients_error']:.1f}")
        
        if 'mean_n_events_error' in metrics:
            text_lines.append(f"Event Count Error: {metrics['mean_n_events_error']:.1f}")
        
        if 'mean_censoring_rate_error' in metrics:
            text_lines.append(f"Censoring Rate Error: {metrics['mean_censoring_rate_error']:.4f}")
        
        text_lines.append("")
        text_lines.append("KM-GPT BENCHMARKS")
        text_lines.append("-" * 40)
        text_lines.append("Median IAE: 0.018")
        text_lines.append("95% CI: 0.002 - 0.088")
        
        # Display text
        y_pos = 0.95
        for line in text_lines:
            if line.startswith("=") or line.startswith("-"):
                ax.text(0.05, y_pos, line, fontsize=9, family='monospace',
                       verticalalignment='top')
            elif line.startswith("  [PASS]") or line.startswith("  [WARN]"):
                ax.text(0.05, y_pos, line, fontsize=9,
                       color='green' if '[PASS]' in line else 'orange',
                       verticalalignment='top', fontweight='bold')
            elif any(line.startswith(x) for x in ["RECONSTRUCTION", "KM-GPT"]):
                ax.text(0.05, y_pos, line, fontsize=10, fontweight='bold',
                       verticalalignment='top')
            else:
                ax.text(0.05, y_pos, line, fontsize=9,
                       verticalalignment='top')
            y_pos -= 0.05
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _real_to_pixel(self, time: float, prob: float, 
                      axis_config: AxisConfig) -> Tuple[int, int]:
        """Convert real-world coordinates to pixel coordinates"""
        cfg = axis_config
        
        # X coordinate (time)
        if cfg.x_end - cfg.x_start != 0:
            x_px = cfg.x_start_px + (time - cfg.x_start) / (cfg.x_end - cfg.x_start) * \
                   (cfg.x_end_px - cfg.x_start_px)
        else:
            x_px = cfg.x_start_px
        
        # Y coordinate (probability) - inverted
        if cfg.y_end - cfg.y_start != 0:
            y_px = cfg.y_end_px + (cfg.y_end - prob) / (cfg.y_end - cfg.y_start) * \
                   (cfg.y_start_px - cfg.y_end_px)
        else:
            y_px = cfg.y_end_px
        
        return int(x_px), int(y_px)
    
    def _compute_km_curve(self, ipd: List[IPDRecord]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert IPD to KM curve"""
        if not ipd:
            return np.array([]), np.array([])
        
        sorted_ipd = sorted(ipd, key=lambda x: x.time)
        
        times = [0.0]
        probs = [1.0]
        
        n_at_risk = len(sorted_ipd)
        survival_prob = 1.0
        
        i = 0
        while i < len(sorted_ipd):
            current_time = sorted_ipd[i].time
            
            n_events = 0
            j = i
            while j < len(sorted_ipd) and sorted_ipd[j].time == current_time:
                if sorted_ipd[j].event == 1:
                    n_events += 1
                j += 1
            
            if n_at_risk > 0 and n_events > 0:
                survival_prob *= (1 - n_events / n_at_risk)
                times.append(current_time)
                probs.append(survival_prob)
            
            n_at_risk -= (j - i)
            i = j
        
        return np.array(times), np.array(probs)
    
    def create_summary_visualization(self, evaluation_results: List[Dict]):
        """
        Create summary visualization showing multiple test cases
        """
        # Select interesting cases (up to 9)
        successful = [r for r in evaluation_results if r.get('success', False)]
        selected = successful[:min(9, len(successful))]
        
        if not selected:
            print("No successful results to visualize")
            return
        
        n_cases = len(selected)
        n_cols = 3
        n_rows = (n_cases + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, result in enumerate(selected):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Extract data
            test_case = result.get('result', {})
            curves = test_case.get('curves', [])
            ipd = test_case.get('ipd', {})
            
            # Plot comparison
            colors = ['blue', 'red', 'green', 'orange']
            
            for i, curve in enumerate(curves):
                if not curve:
                    continue
                
                times = [t for t, _ in curve]
                probs = [p for _, p in curve]
                ax.plot(times, probs, color=colors[i % len(colors)],
                       linewidth=2, label=f'Original', alpha=0.7)
                
                group_name = f"Group_{i+1}"
                if group_name in ipd:
                    recon_times, recon_probs = self._compute_km_curve(ipd[group_name])
                    ax.plot(recon_times, recon_probs, color=colors[i % len(colors)],
                           linewidth=2, linestyle='--', label=f'Reconstructed', alpha=0.7)
            
            # Add metrics to title
            if result.get('aggregate_metrics', {}).get('success', False):
                metrics = result['aggregate_metrics']
                title = f"{result['test_name']}\nIAE={metrics['mean_iae']:.4f}"
            else:
                title = result['test_name']
            
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('Time', fontsize=8)
            ax.set_ylabel('Survival', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
        
        # Hide empty subplots
        for idx in range(n_cases, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        fig.suptitle('IPD Recovery Pipeline - Summary of Test Cases', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'summary_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Summary visualization saved to {save_path}")
        
        plt.close(fig)


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    print("Reconstruction Visualization Tool")
    print("=" * 60)
    
    # Example: visualize a single reconstruction
    from pipeline.synthetic import SyntheticKMGenerator, SyntheticConfig
    from pipeline.core import IPDRecoveryPipeline, RiskTable
    
    # Generate synthetic case
    config = SyntheticConfig(
        n_patients=100,
        median_survival=24.0,
        censoring_rate=0.3,
        n_groups=2,
        hazard_ratio=0.7
    )
    
    generator = SyntheticKMGenerator(config)
    test_case = generator.plot_km_curve('test_km.png')
    
    # Run pipeline
    pipeline = IPDRecoveryPipeline(n_curves=2, use_preprocessing=False, use_ocr=False)
    
    manual_risk_table = RiskTable(
        timepoints=test_case['risk_tables'][0]['timepoints'],
        n_at_risk={
            'Group_1': test_case['risk_tables'][0]['n_at_risk'],
            'Group_2': test_case['risk_tables'][1]['n_at_risk']
        },
        group_labels=['Group_1', 'Group_2']
    )
    
    result = pipeline.process(
        img=test_case['image'],
        mask=test_case['mask'],
        manual_axis_ranges={'x_range': (0, config.follow_up_time), 'y_range': (0, 1.0)},
        manual_risk_table=manual_risk_table
    )
    
    # Visualize
    visualizer = ReconstructionVisualizer(output_dir='./visualizations')
    visualizer.visualize_reconstruction(
        img=test_case['image'],
        original_curves=test_case['curves'],
        reconstructed_ipd=result['ipd'],
        axis_config=result['axis_config'],
        test_name='example_reconstruction',
        metrics=result['validation']
    )
    
    print("\nVisualization complete!")