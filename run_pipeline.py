#!/usr/bin/env python3
"""
Main Runner Script for IPD Recovery Pipeline
Provides command-line interface and automated workflows
"""

import argparse
import os
import sys
import numpy as np
import cv2
import json
from pathlib import Path

from pipeline.core import IPDRecoveryPipeline, RiskTable
from pipeline.synthetic import SyntheticTestSuite, SyntheticKMGenerator, SyntheticConfig
from pipeline.evaluation import IPDEvaluator
from pipeline.visualization import ReconstructionVisualizer


def run_single_reconstruction(args):
    """Run IPD reconstruction on a single KM plot"""
    print("="*80)
    print("IPD RECOVERY PIPELINE - Single Image Mode")
    print("="*80)
    
    # Load image
    print(f"\nLoading image: {args.image}")
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not load image {args.image}")
        return 1
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load mask
    if args.mask:
        print(f"Loading mask: {args.mask}")
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    else:
        print("No mask provided - creating from colored regions...")
        # Auto-generate mask from colored regions
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = img_hsv[:, :, 1]
        value = img_hsv[:, :, 2]
        # Require saturation > 50 to exclude gray/black text and axis lines
        # Exclude very dark pixels (text) but allow bright colored pixels
        mask = ((saturation > 50) & (value > 40)).astype(np.uint8) * 255
        # Morphological opening to remove small noise dots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Initialize pipeline
    print(f"\nInitializing pipeline (n_curves={args.n_curves})...")
    pipeline = IPDRecoveryPipeline(
        n_curves=args.n_curves,
        use_preprocessing=args.preprocess,
        use_ocr=args.use_ocr
    )
    
    # Manual configuration
    manual_ranges = None
    manual_risk_table = None
    
    if args.config:
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        if 'axis_ranges' in config:
            manual_ranges = config['axis_ranges']
        
        if 'risk_table' in config:
            rt = config['risk_table']
            manual_risk_table = RiskTable(
                timepoints=rt['timepoints'],
                n_at_risk=rt['n_at_risk'],
                group_labels=rt.get('group_labels', [])
            )
    
    # Run pipeline
    print("\nRunning IPD reconstruction...")
    try:
        result = pipeline.process(
            img=img_rgb,
            mask=mask,
            manual_axis_ranges=manual_ranges,
            manual_risk_table=manual_risk_table
        )
        print("[OK] Reconstruction successful!")
    except Exception as e:
        print(f"[FAIL] Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Display results
    print("\nResults:")
    print("-" * 80)
    for group, ipd in result['ipd'].items():
        print(f"\n{group}:")
        print(f"  Total patients: {len(ipd)}")
        print(f"  Events: {sum(1 for r in ipd if r.event == 1)}")
        print(f"  Censored: {sum(1 for r in ipd if r.event == 0)}")
        
        if group in result['validation']:
            val = result['validation'][group]
            if val.get('valid', False):
                print(f"  IAE: {val['iae']:.6f}")
                print(f"  Median survival (original): {val['median_original']:.2f}")
                print(f"  Median survival (reconstructed): {val['median_reconstructed']:.2f}")
    
    # Save outputs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export IPD
    ipd_path = output_dir / "reconstructed_ipd.csv"
    pipeline.export_ipd_to_csv(result['ipd'], str(ipd_path))
    print(f"\n[OK] IPD exported to {ipd_path}")
    
    # Visualize
    viz_path = output_dir / "validation.png"
    pipeline.visualize_results(img_rgb, result, save_path=str(viz_path))
    print(f"[OK] Validation plot saved to {viz_path}")
    
    # Save detailed visualization if requested
    if args.detailed_viz:
        visualizer = ReconstructionVisualizer(output_dir=str(output_dir))
        metrics = result['validation'].get(list(result['ipd'].keys())[0], {})
        
        visualizer.visualize_reconstruction(
            img=img_rgb,
            original_curves=result['curves'],
            reconstructed_ipd=result['ipd'],
            axis_config=result['axis_config'],
            test_name='reconstruction',
            metrics={'success': True, **metrics} if metrics else None
        )
    
    print("\n" + "="*80)
    print("Reconstruction complete!")
    print("="*80)
    
    return 0


def run_evaluation(args):
    """Run comprehensive evaluation on synthetic data"""
    print("="*80)
    print("IPD RECOVERY PIPELINE - Evaluation Mode")
    print("="*80)
    
    # Generate synthetic test data
    print("\nStep 1: Generating synthetic test battery...")
    suite = SyntheticTestSuite(output_dir=args.synthetic_dir)
    test_results = suite.generate_test_battery()
    suite.create_summary_report(test_results)
    print(f"[OK] Generated {len(test_results)} test cases")
    
    # Initialize pipeline
    print("\nStep 2: Initializing pipeline...")
    pipeline = IPDRecoveryPipeline(
        n_curves=1,  # Most synthetic cases are single-arm
        use_preprocessing=args.preprocess,
        use_ocr=False  # Use manual config for controlled evaluation
    )
    print("[OK] Pipeline initialized")
    
    # Run evaluation
    print("\nStep 3: Running comprehensive evaluation...")
    evaluator = IPDEvaluator(output_dir=args.output)
    evaluation_results = evaluator.evaluate_test_suite(test_results, pipeline)
    
    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    successful = [r for r in evaluation_results if r['success'] and 
                 r.get('aggregate_metrics', {}).get('success', False)]
    
    if successful:
        iaes = [r['aggregate_metrics']['mean_iae'] for r in successful]
        rmses = [r['aggregate_metrics']['mean_rmse'] for r in successful]
        
        print(f"\nSuccess rate: {len(successful)}/{len(evaluation_results)} "
              f"({len(successful)/len(evaluation_results)*100:.1f}%)")
        print(f"\nMedian IAE: {np.median(iaes):.6f}")
        print(f"  KM-GPT benchmark: 0.018 (95% CI: 0.002-0.088)")
        print(f"\nMedian RMSE: {np.median(rmses):.6f}")
        print(f"\n95th percentile IAE: {np.percentile(iaes, 95):.6f}")
        print(f"  KM-GPT benchmark 95th percentile: 0.088")
        
        # Performance assessment
        if np.median(iaes) <= 0.018:
            print("\n[PASS][PASS] Performance EXCEEDS KM-GPT median benchmark!")
        elif np.median(iaes) <= 0.088:
            print("\n[OK] Performance is within KM-GPT's 95% confidence interval")
        else:
            print("\n[WARN] Performance is below KM-GPT benchmarks")
    else:
        print("\n[WARN] No successful evaluations")
    
    print(f"\nDetailed results saved to: {args.output}")
    print("="*80)
    
    return 0


def run_demo(args):
    """Run demonstration with example synthetic case"""
    print("="*80)
    print("IPD RECOVERY PIPELINE - Demo Mode")
    print("="*80)
    
    # Generate example case
    print("\nGenerating example KM plot...")
    config = SyntheticConfig(
        n_patients=120,
        median_survival=24.0,
        censoring_rate=0.3,
        n_groups=2,
        hazard_ratio=0.7,
        resolution=(1000, 750),
        include_risk_table=True
    )
    
    generator = SyntheticKMGenerator(config)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / 'demo_km_plot.png'
    test_case = generator.plot_km_curve(str(save_path))
    print(f"[OK] KM plot saved to {save_path}")
    
    # Save ground truth
    for i, ipd in enumerate(test_case['ipd']):
        ipd_path = output_dir / f'demo_ground_truth_group{i+1}.csv'
        ipd.to_csv(ipd_path, index=False)
    print(f"[OK] Ground truth IPD saved")
    
    # Run pipeline
    print("\nRunning IPD reconstruction...")
    pipeline = IPDRecoveryPipeline(n_curves=2, use_preprocessing=False, use_ocr=False)
    
    manual_risk_table = RiskTable(
        timepoints=test_case['risk_tables'][0]['timepoints'],
        n_at_risk={
            'Group_1': test_case['risk_tables'][0]['n_at_risk'],
            'Group_2': test_case['risk_tables'][1]['n_at_risk']
        },
        group_labels=['Treatment', 'Control']
    )
    
    result = pipeline.process(
        img=test_case['image'],
        mask=test_case['mask'],
        manual_axis_ranges={'x_range': (0, config.follow_up_time), 'y_range': (0, 1.0)},
        manual_risk_table=manual_risk_table
    )
    
    # Export
    ipd_path = output_dir / 'demo_reconstructed_ipd.csv'
    pipeline.export_ipd_to_csv(result['ipd'], str(ipd_path))
    print(f"[OK] Reconstructed IPD saved to {ipd_path}")
    
    # Visualize
    viz_path = output_dir / 'demo_validation.png'
    pipeline.visualize_results(test_case['image'], result, save_path=str(viz_path))
    
    # Detailed visualization
    visualizer = ReconstructionVisualizer(output_dir=str(output_dir))
    visualizer.visualize_reconstruction(
        img=test_case['image'],
        original_curves=test_case['curves'],
        reconstructed_ipd=result['ipd'],
        axis_config=result['axis_config'],
        test_name='demo',
        metrics=result['validation'].get('Group_1', {})
    )
    
    # Print results
    print("\nResults:")
    print("-" * 80)
    for group, ipd_list in result['ipd'].items():
        print(f"\n{group}:")
        print(f"  Patients: {len(ipd_list)}")
        print(f"  Events: {sum(1 for r in ipd_list if r.event == 1)}")
        
        if group in result['validation']:
            val = result['validation'][group]
            print(f"  IAE: {val.get('iae', 'N/A'):.6f}")
    
    print(f"\n[OK] All outputs saved to {output_dir}")
    print("="*80)
    
    return 0


def run_analyze(args):
    """Analyze a KM plot using GPT-4o vision (MMPU)"""
    print("="*80)
    print("IPD RECOVERY PIPELINE - MMPU Analysis Mode")
    print("="*80)

    try:
        from pipeline.mmpu import analyze_km_plot
    except ImportError:
        print("[FAIL] openai package required. Install with: pip install openai")
        return 1

    config = analyze_km_plot(args.image, verbose=True)

    # Save config as JSON
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "mmpu_config.json"
        # Convert config to serializable form
        serializable = {
            'axis_ranges': config['axis_ranges'],
            'n_curves': config['n_curves'],
        }
        if 'group_labels' in config:
            serializable['group_labels'] = config['group_labels']
        if 'risk_table' in config:
            rt = config['risk_table']
            serializable['risk_table'] = {
                'timepoints': rt.timepoints,
                'n_at_risk': rt.n_at_risk,
                'group_labels': rt.group_labels
            }
        with open(config_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\n[OK] Configuration saved to {config_path}")

    # Optionally run reconstruction with extracted config
    if args.reconstruct:
        print("\nRunning reconstruction with MMPU-extracted configuration...")
        img = cv2.imread(args.image)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Auto-generate mask
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = img_hsv[:, :, 1]
        mask = (saturation > 30).astype(np.uint8) * 255

        pipeline = IPDRecoveryPipeline(
            n_curves=config['n_curves'],
            use_preprocessing=True,
            use_ocr=False
        )

        result = pipeline.process(
            img=img_rgb,
            mask=mask,
            manual_axis_ranges=config['axis_ranges'],
            manual_risk_table=config.get('risk_table')
        )

        output_dir = Path(args.output or './output/analyze')
        output_dir.mkdir(parents=True, exist_ok=True)
        ipd_path = output_dir / "reconstructed_ipd.csv"
        pipeline.export_ipd_to_csv(result['ipd'], str(ipd_path))
        print(f"[OK] IPD exported to {ipd_path}")

    return 0


def run_powered_evaluation(args):
    """Run statistically powered evaluation with bootstrap CIs"""
    print("="*80)
    print("IPD RECOVERY PIPELINE - Powered Evaluation Mode")
    print("="*80)

    from pipeline.powered_evaluation import run_powered_evaluation as _run_powered

    _run_powered(
        n_seeds=args.seeds,
        output_dir=args.output,
    )

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='IPD Recovery Pipeline - Reconstruct individual patient data from KM plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo
  python run_pipeline.py demo --output ./output/demo

  # Reconstruct from single image
  python run_pipeline.py reconstruct --image km_plot.png --n-curves 2 --output ./results

  # Run full evaluation
  python run_pipeline.py evaluate --output ./evaluation

  # Analyze image with GPT-4o vision (MMPU)
  python run_pipeline.py analyze --image km_plot.png --output ./output/analyze

  # Run powered evaluation (50 seeds with bootstrap CIs)
  python run_pipeline.py powered-evaluate --seeds 50
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Reconstruct command
    reconstruct_parser = subparsers.add_parser('reconstruct', help='Reconstruct IPD from single image')
    reconstruct_parser.add_argument('--image', required=True, help='Path to KM plot image')
    reconstruct_parser.add_argument('--mask', help='Path to segmentation mask (optional)')
    reconstruct_parser.add_argument('--config', help='Path to JSON config file')
    reconstruct_parser.add_argument('--n-curves', type=int, default=1, help='Number of curves')
    reconstruct_parser.add_argument('--preprocess', action='store_true', help='Apply preprocessing')
    reconstruct_parser.add_argument('--use-ocr', action='store_true', help='Use OCR for extraction')
    reconstruct_parser.add_argument('--detailed-viz', action='store_true', help='Create detailed visualization')
    reconstruct_parser.add_argument('--output', default='./output', help='Output directory')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run comprehensive evaluation')
    eval_parser.add_argument('--synthetic-dir', default='./output/synthetic_data', help='Directory for synthetic data')
    eval_parser.add_argument('--output', default='./output/evaluation', help='Output directory')
    eval_parser.add_argument('--preprocess', action='store_true', help='Apply preprocessing')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')
    demo_parser.add_argument('--output', default='./output/demo', help='Output directory')

    # Analyze command (MMPU)
    analyze_parser = subparsers.add_parser('analyze', help='Analyze KM plot with GPT-4o vision (MMPU)')
    analyze_parser.add_argument('--image', required=True, help='Path to KM plot image')
    analyze_parser.add_argument('--reconstruct', action='store_true', help='Also run reconstruction')
    analyze_parser.add_argument('--output', default='./output/analyze', help='Output directory')

    # Powered evaluation command
    powered_parser = subparsers.add_parser('powered-evaluate',
                                           help='Run powered evaluation with bootstrap CIs')
    powered_parser.add_argument('--seeds', type=int, default=50, help='Number of random seeds')
    powered_parser.add_argument('--output', default='./output/powered_evaluation',
                                help='Output directory')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Route to appropriate function
    if args.command == 'reconstruct':
        return run_single_reconstruction(args)
    elif args.command == 'evaluate':
        return run_evaluation(args)
    elif args.command == 'demo':
        return run_demo(args)
    elif args.command == 'analyze':
        return run_analyze(args)
    elif args.command == 'powered-evaluate':
        return run_powered_evaluation(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())