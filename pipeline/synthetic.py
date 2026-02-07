"""
Synthetic KM Plot Generator for IPD Recovery Pipeline Evaluation
Creates realistic KM plots with known ground truth for benchmarking

Generates edge cases:
- Varying resolutions (low to high)
- Multiple overlapping curves
- Different font styles in risk tables
- Truncated y-axis
- Heavy/light censoring
- Various sample sizes and survival distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from typing import List, Tuple, Dict, Optional
import cv2
from dataclasses import dataclass
import pandas as pd
import os


@dataclass
class SyntheticConfig:
    """Configuration for generating synthetic KM plots"""
    n_patients: int = 100
    median_survival: float = 24.0  # months
    censoring_rate: float = 0.3
    follow_up_time: float = 60.0  # months
    n_groups: int = 1
    hazard_ratio: float = 1.0  # For multi-arm trials
    resolution: Tuple[int, int] = (800, 600)  # (width, height)
    dpi: int = 100
    include_risk_table: bool = True
    truncate_y_axis: bool = False
    y_axis_min: float = 0.0
    font_style: str = 'default'  # 'default', 'serif', 'sans-serif'
    add_noise: bool = False
    noise_level: float = 0.02


class SyntheticKMGenerator:
    """Generate synthetic KM plots with known ground truth IPD"""
    
    def __init__(self, config: SyntheticConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        np.random.seed(seed)
    
    def generate_ipd(self, group_id: int = 0) -> pd.DataFrame:
        """
        Generate synthetic IPD (ground truth)
        
        Args:
            group_id: Group identifier for multi-arm trials
            
        Returns:
            DataFrame with columns: time, event, group
        """
        cfg = self.config
        
        # Apply hazard ratio for treatment groups
        effective_median = cfg.median_survival * (cfg.hazard_ratio ** group_id)
        
        # Generate survival times from exponential distribution
        # Lambda = log(2) / median
        lambda_param = np.log(2) / effective_median
        survival_times = expon.rvs(scale=1/lambda_param, size=cfg.n_patients)
        
        # Generate censoring times
        n_censored = int(cfg.n_patients * cfg.censoring_rate)
        censored_indices = np.random.choice(cfg.n_patients, n_censored, replace=False)
        
        events = np.ones(cfg.n_patients, dtype=int)
        events[censored_indices] = 0
        
        # For censored patients, sample censoring time uniformly
        for idx in censored_indices:
            censoring_time = np.random.uniform(0, min(survival_times[idx], cfg.follow_up_time))
            survival_times[idx] = censoring_time
        
        # Truncate at follow-up time
        survival_times = np.minimum(survival_times, cfg.follow_up_time)
        
        # Create DataFrame
        ipd = pd.DataFrame({
            'time': survival_times,
            'event': events,
            'group': f'Group_{group_id + 1}'
        })
        
        return ipd
    
    def compute_km_curve(self, ipd: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Kaplan-Meier curve from IPD
        
        Returns:
            times, survival_probs arrays
        """
        # Sort by time
        ipd_sorted = ipd.sort_values('time')
        
        times = [0.0]
        survival_probs = [1.0]
        
        n_at_risk = len(ipd)
        current_survival = 1.0
        
        i = 0
        while i < len(ipd_sorted):
            current_time = ipd_sorted.iloc[i]['time']
            
            # Count events at this exact time
            n_events = 0
            j = i
            while j < len(ipd_sorted) and ipd_sorted.iloc[j]['time'] == current_time:
                if ipd_sorted.iloc[j]['event'] == 1:
                    n_events += 1
                j += 1
            
            # Update survival probability
            if n_at_risk > 0 and n_events > 0:
                current_survival *= (1 - n_events / n_at_risk)
                times.append(current_time)
                survival_probs.append(current_survival)
            
            # Update number at risk
            n_at_risk -= (j - i)
            i = j
        
        return np.array(times), np.array(survival_probs)
    
    def generate_risk_table_data(self, ipd: pd.DataFrame, 
                                 time_intervals: Optional[List[float]] = None) -> Dict:
        """
        Generate number at risk table from IPD
        
        Args:
            ipd: Individual patient data
            time_intervals: Time points for risk table (if None, use defaults)
            
        Returns:
            Dict with timepoints and n_at_risk
        """
        if time_intervals is None:
            # Default: every 12 months up to follow-up time
            time_intervals = list(range(0, int(self.config.follow_up_time) + 1, 12))
        
        n_at_risk = []
        for t in time_intervals:
            # Count patients still at risk at time t
            at_risk = ipd[ipd['time'] >= t]
            n_at_risk.append(len(at_risk))
        
        return {
            'timepoints': time_intervals,
            'n_at_risk': n_at_risk
        }
    
    def plot_km_curve(self, save_path: str = 'synthetic_km.png') -> Dict:
        """
        Generate complete KM plot with risk table
        
        Returns:
            Dict with ground truth data and plot metadata
        """
        cfg = self.config
        
        # Generate IPD for all groups
        all_ipd = []
        all_curves = []
        all_risk_tables = []
        
        for group_id in range(cfg.n_groups):
            ipd = self.generate_ipd(group_id)
            all_ipd.append(ipd)
            
            times, survival_probs = self.compute_km_curve(ipd)
            all_curves.append((times, survival_probs))
            
            risk_table = self.generate_risk_table_data(ipd)
            all_risk_tables.append(risk_table)
        
        # Create figure
        fig_width = cfg.resolution[0] / cfg.dpi
        fig_height = cfg.resolution[1] / cfg.dpi
        
        if cfg.include_risk_table:
            fig, (ax_main, ax_risk) = plt.subplots(
                2, 1, figsize=(fig_width, fig_height),
                gridspec_kw={'height_ratios': [4, 1]},
                dpi=cfg.dpi
            )
        else:
            fig, ax_main = plt.subplots(figsize=(fig_width, fig_height), dpi=cfg.dpi)
            ax_risk = None
        
        # Plot KM curves
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for group_id, (times, survival_probs) in enumerate(all_curves):
            label = f'Group {group_id + 1}' if cfg.n_groups > 1 else 'Treatment'
            ax_main.step(times, survival_probs, where='post', 
                        color=colors[group_id % len(colors)],
                        linewidth=2, label=label)
            
            # Add censoring marks
            ipd = all_ipd[group_id]
            censored = ipd[ipd['event'] == 0]
            if len(censored) > 0:
                # Get survival probability at censoring times
                censor_times = censored['time'].values
                censor_probs = []
                for ct in censor_times:
                    # Find survival prob at this time
                    idx = np.searchsorted(times, ct, side='right') - 1
                    idx = max(0, min(idx, len(survival_probs) - 1))
                    censor_probs.append(survival_probs[idx])
                
                ax_main.scatter(censor_times, censor_probs, 
                              marker='+', s=100, 
                              color=colors[group_id % len(colors)],
                              linewidths=2, zorder=10)
        
        # Configure main axis
        y_min = cfg.y_axis_min if cfg.truncate_y_axis else 0.0
        ax_main.set_xlim(0, cfg.follow_up_time)
        ax_main.set_ylim(y_min, 1.0)
        ax_main.set_xlabel('Time (months)', fontsize=12)
        ax_main.set_ylabel('Survival Probability', fontsize=12)
        ax_main.set_title('Kaplan-Meier Survival Curve', fontsize=14, fontweight='bold')
        ax_main.grid(True, alpha=0.3, linestyle='--')
        ax_main.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Add risk table if requested
        if cfg.include_risk_table and ax_risk is not None:
            self._add_risk_table(ax_risk, all_risk_tables, all_ipd, colors)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=cfg.dpi, bbox_inches='tight', facecolor='white')
        
        # Get axis positions in pixels BEFORE converting to array
        # This gives us the actual plot boundaries
        ax_bbox = ax_main.get_position()
        fig_width_px = fig.get_figwidth() * cfg.dpi
        fig_height_px = fig.get_figheight() * cfg.dpi
        
        # Convert normalized axis coordinates to pixels
        x_start_px = int(ax_bbox.x0 * fig_width_px)
        x_end_px = int(ax_bbox.x1 * fig_width_px)
        y_end_px = int((1 - ax_bbox.y1) * fig_height_px)  # Top (inverted y)
        y_start_px = int((1 - ax_bbox.y0) * fig_height_px)  # Bottom (inverted y)
        
        axis_positions = {
            'x_start_px': x_start_px,
            'x_end_px': x_end_px,
            'y_start_px': y_start_px,
            'y_end_px': y_end_px,
            'x_range': (0, cfg.follow_up_time),
            'y_range': (cfg.y_axis_min if cfg.truncate_y_axis else 0.0, 1.0)
        }
        
        # Also save as array for processing
        fig.canvas.draw()
        # Use buffer_rgba() which works in newer matplotlib versions
        try:
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # Newer matplotlib versions
            img_array = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        
        plt.close(fig)
        
        # Add noise if requested
        if cfg.add_noise:
            img_array = self._add_noise(img_array)
            cv2.imwrite(save_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        # Create segmentation mask (curve pixels only)
        mask = self._create_curve_mask(img_array, all_curves, cfg)
        
        return {
            'ipd': all_ipd,
            'curves': all_curves,
            'risk_tables': all_risk_tables,
            'image': img_array,
            'mask': mask,
            'config': cfg,
            'axis_positions': axis_positions
        }
    
    def _add_risk_table(self, ax, risk_tables, all_ipd, colors):
        """Add number at risk table below main plot"""
        ax.axis('off')
        
        # Get timepoints (assume same for all groups)
        timepoints = risk_tables[0]['timepoints']
        
        # Create table
        table_data = []
        row_labels = []
        
        for group_id, (risk_table, ipd) in enumerate(zip(risk_tables, all_ipd)):
            group_name = ipd['group'].iloc[0]
            row_labels.append(group_name)
            table_data.append(risk_table['n_at_risk'])
        
        # Create text-based table
        y_pos = 0.8
        ax.text(0.0, y_pos, 'Number at risk', fontsize=10, fontweight='bold',
               verticalalignment='top')
        
        y_pos -= 0.15
        for group_id, (label, counts) in enumerate(zip(row_labels, table_data)):
            # Group label
            ax.text(0.0, y_pos, label, fontsize=9, 
                   color=colors[group_id % len(colors)],
                   verticalalignment='top')
            
            # Numbers
            x_positions = np.linspace(0.15, 0.95, len(timepoints))
            for x, count in zip(x_positions, counts):
                ax.text(x, y_pos, str(count), fontsize=9,
                       horizontalalignment='center',
                       verticalalignment='top')
            
            y_pos -= 0.15
        
        # Add time labels at top
        y_pos = 0.95
        x_positions = np.linspace(0.15, 0.95, len(timepoints))
        for x, t in zip(x_positions, timepoints):
            ax.text(x, y_pos, str(int(t)), fontsize=9,
                   horizontalalignment='center',
                   verticalalignment='top',
                   fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _create_curve_mask(self, img_array: np.ndarray, 
                          curves: List[Tuple[np.ndarray, np.ndarray]],
                          cfg: SyntheticConfig) -> np.ndarray:
        """
        Create binary segmentation mask for curve pixels
        Uses color detection focused on the KM curve colors
        """
        h, w = img_array.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Standard matplotlib colors used for curves
        # '#1f77b4' (blue), '#ff7f0e' (orange), '#2ca02c' (green), etc.
        curve_colors_rgb = [
            (31, 119, 180),   # Blue
            (255, 127, 14),   # Orange  
            (44, 160, 44),    # Green
            (214, 39, 40),    # Red
            (148, 103, 189),  # Purple
        ]
        
        # Convert to HSV for better color matching
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Create mask for each curve color
        for rgb_color in curve_colors_rgb[:cfg.n_groups]:
            # Convert RGB to HSV
            rgb_img = np.uint8([[rgb_color]])
            hsv_color = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)[0][0]
            
            # Create range around the color (convert to int to avoid overflow)
            h_range = 15  # Hue tolerance
            s_range = 50  # Saturation tolerance
            v_range = 50  # Value tolerance
            
            h, s, v = int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])
            
            lower = np.array([max(0, h - h_range), 
                             max(0, s - s_range),
                             max(0, v - v_range)])
            upper = np.array([min(180, h + h_range),
                             min(255, s + s_range),
                             min(255, v + v_range)])
            
            color_mask = cv2.inRange(img_hsv, lower, upper)
            mask = cv2.bitwise_or(mask, color_mask)
        
        # Also detect the censoring marks (+ symbols) which are the same colors
        # They should already be captured by the color mask
        
        # Clean up: remove very small components (noise)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Fill small gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Clean up mask - but less aggressively
        kernel = np.ones((2, 2), np.uint8)  # Smaller kernel
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Remove very small noise
        kernel_small = np.ones((1, 1), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        return mask
    
    def _add_noise(self, img: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, self.config.noise_level * 255, img.shape)
        noisy_img = img.astype(np.float32) + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return noisy_img


class SyntheticTestSuite:
    """
    Generate comprehensive test suite with various edge cases
    """
    
    def __init__(self, output_dir: str = './output/synthetic_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_test_battery(self, seed: int = 42) -> List[Dict]:
        """
        Generate comprehensive test battery covering edge cases

        Args:
            seed: Random seed for reproducibility

        Returns:
            List of test cases with ground truth
        """
        test_cases = []
        
        # Test 1: Basic single-arm trial
        test_cases.append({
            'name': 'basic_single_arm',
            'config': SyntheticConfig(
                n_patients=100,
                median_survival=24.0,
                censoring_rate=0.3,
                resolution=(800, 600)
            )
        })
        
        # Test 2: Low resolution
        test_cases.append({
            'name': 'low_resolution',
            'config': SyntheticConfig(
                n_patients=100,
                median_survival=24.0,
                censoring_rate=0.3,
                resolution=(400, 300),
                dpi=50
            )
        })
        
        # Test 3: High resolution
        test_cases.append({
            'name': 'high_resolution',
            'config': SyntheticConfig(
                n_patients=100,
                median_survival=24.0,
                censoring_rate=0.3,
                resolution=(1600, 1200),
                dpi=150
            )
        })
        
        # Test 4: Multiple overlapping curves
        test_cases.append({
            'name': 'overlapping_curves',
            'config': SyntheticConfig(
                n_patients=150,
                median_survival=24.0,
                censoring_rate=0.25,
                n_groups=3,
                hazard_ratio=0.85,  # Small difference between groups
                resolution=(800, 600)
            )
        })
        
        # Test 5: Truncated y-axis
        test_cases.append({
            'name': 'truncated_y_axis',
            'config': SyntheticConfig(
                n_patients=100,
                median_survival=24.0,
                censoring_rate=0.3,
                truncate_y_axis=True,
                y_axis_min=0.2,
                resolution=(800, 600)
            )
        })
        
        # Test 6: Heavy censoring
        test_cases.append({
            'name': 'heavy_censoring',
            'config': SyntheticConfig(
                n_patients=100,
                median_survival=24.0,
                censoring_rate=0.7,
                resolution=(800, 600)
            )
        })
        
        # Test 7: Light censoring
        test_cases.append({
            'name': 'light_censoring',
            'config': SyntheticConfig(
                n_patients=100,
                median_survival=24.0,
                censoring_rate=0.05,
                resolution=(800, 600)
            )
        })
        
        # Test 8: Small sample size
        test_cases.append({
            'name': 'small_sample',
            'config': SyntheticConfig(
                n_patients=30,
                median_survival=24.0,
                censoring_rate=0.3,
                resolution=(800, 600)
            )
        })
        
        # Test 9: Large sample size
        test_cases.append({
            'name': 'large_sample',
            'config': SyntheticConfig(
                n_patients=500,
                median_survival=24.0,
                censoring_rate=0.3,
                resolution=(800, 600)
            )
        })
        
        # Test 10: Short follow-up
        test_cases.append({
            'name': 'short_followup',
            'config': SyntheticConfig(
                n_patients=100,
                median_survival=6.0,
                censoring_rate=0.3,
                follow_up_time=12.0,
                resolution=(800, 600)
            )
        })
        
        # Test 11: Long follow-up
        test_cases.append({
            'name': 'long_followup',
            'config': SyntheticConfig(
                n_patients=100,
                median_survival=48.0,
                censoring_rate=0.3,
                follow_up_time=120.0,
                resolution=(800, 600)
            )
        })
        
        # Test 12: Noisy image
        test_cases.append({
            'name': 'noisy_image',
            'config': SyntheticConfig(
                n_patients=100,
                median_survival=24.0,
                censoring_rate=0.3,
                add_noise=True,
                noise_level=0.03,
                resolution=(800, 600)
            )
        })
        
        # Test 13: Two-arm trial
        test_cases.append({
            'name': 'two_arm_trial',
            'config': SyntheticConfig(
                n_patients=120,
                median_survival=20.0,
                censoring_rate=0.3,
                n_groups=2,
                hazard_ratio=0.7,  # 30% risk reduction
                resolution=(800, 600)
            )
        })
        
        # Test 14: No risk table
        test_cases.append({
            'name': 'no_risk_table',
            'config': SyntheticConfig(
                n_patients=100,
                median_survival=24.0,
                censoring_rate=0.3,
                include_risk_table=False,
                resolution=(800, 600)
            )
        })
        
        # Generate all test cases
        results = []
        for test_case in test_cases:
            print(f"Generating {test_case['name']}...")
            
            generator = SyntheticKMGenerator(test_case['config'], seed=seed)
            save_path = os.path.join(self.output_dir, f"{test_case['name']}.png")
            
            result = generator.plot_km_curve(save_path)
            result['name'] = test_case['name']
            result['save_path'] = save_path
            
            # Save ground truth IPD
            for i, ipd in enumerate(result['ipd']):
                ipd_path = os.path.join(self.output_dir, 
                                       f"{test_case['name']}_ipd_group{i+1}.csv")
                ipd.to_csv(ipd_path, index=False)
            
            # Save mask
            mask_path = os.path.join(self.output_dir, f"{test_case['name']}_mask.png")
            cv2.imwrite(mask_path, result['mask'])
            
            results.append(result)
        
        print(f"\nGenerated {len(results)} test cases in {self.output_dir}")
        return results
    
    def create_summary_report(self, results: List[Dict]):
        """Create summary report of test battery"""
        report = []
        report.append("=" * 80)
        report.append("SYNTHETIC TEST BATTERY SUMMARY")
        report.append("=" * 80)
        report.append("")
        
        for result in results:
            cfg = result['config']
            report.append(f"Test: {result['name']}")
            report.append(f"  Resolution: {cfg.resolution[0]}x{cfg.resolution[1]} @ {cfg.dpi} DPI")
            report.append(f"  Groups: {cfg.n_groups}")
            report.append(f"  Patients per group: {cfg.n_patients}")
            report.append(f"  Median survival: {cfg.median_survival} months")
            report.append(f"  Censoring rate: {cfg.censoring_rate * 100:.1f}%")
            report.append(f"  Follow-up: {cfg.follow_up_time} months")
            
            for i, ipd in enumerate(result['ipd']):
                n_events = (ipd['event'] == 1).sum()
                n_censored = (ipd['event'] == 0).sum()
                report.append(f"  Group {i+1}: {n_events} events, {n_censored} censored")
            
            report.append("")
        
        report_path = os.path.join(self.output_dir, 'test_battery_summary.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Summary report saved to {report_path}")


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    print("Synthetic KM Plot Generator")
    print("=" * 60)
    
    # Generate comprehensive test battery
    suite = SyntheticTestSuite(output_dir='./output/synthetic_data')
    results = suite.generate_test_battery()
    suite.create_summary_report(results)
    
    print("\nTest battery generation complete!")
    print(f"Generated {len(results)} test cases")
    print("\nExample test case details:")
    
    # Show details of first test case
    example = results[0]
    print(f"  Name: {example['name']}")
    print(f"  Image shape: {example['image'].shape}")
    print(f"  Mask shape: {example['mask'].shape}")
    print(f"  Number of curves: {len(example['curves'])}")
    
    if example['ipd']:
        ipd = example['ipd'][0]
        print(f"  IPD records: {len(ipd)}")
        print(f"  Events: {(ipd['event'] == 1).sum()}")
        print(f"  Censored: {(ipd['event'] == 0).sum()}")