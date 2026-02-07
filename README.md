# KM Curve Digitization Pipeline

A pipeline for extracting Individual Patient Data (IPD) from Kaplan-Meier survival curve images. Based on the KM-GPT architecture described in [arxiv:2509.18141v1](https://arxiv.org/abs/2509.18141v1).

## Overview

This tool takes a KM plot image and reconstructs the underlying patient-level data (time-to-event, event indicator, treatment group) that produced the curve. This enables meta-analyses from published survival curves where raw data is unavailable.

## Pipeline Architecture

The pipeline follows five stages:

### Stage 1: Image Preprocessing
- Adaptive resolution enhancement (2x upscale for low-res images, skipped for >1200px)
- Laplacian sharpening and non-local means denoising (skipped for clean high-res images)
- Mask resizing to match preprocessed image dimensions

### Stage 2: Axis Detection & Calibration
- Detects x-axis and y-axis lines by finding the longest horizontal/vertical dark pixel runs
- Supports OCR-based tick label extraction (requires Tesseract) or manual axis range specification
- Converts pixel coordinates to real-world values (time, survival probability)

### Stage 3: Curve Extraction & Differentiation
- Filters mask pixels to the detected plot area (removes risk table contamination)
- Single curve: median-y per x-column extraction with adaptive step enforcement
- Multiple curves: KMeans clustering in HSL color space with k-NN refinement for overlapping regions

### Stage 4: Pixel-to-Scale Calibration
- Linear transformation from pixel coordinates to real-world values
- Uses axis endpoints and tick labels from Stage 2
- Enforces monotone non-increasing step function structure (KM property)

### Stage 5: IPD Reconstruction (iKM Algorithm)
- Implements Guyot et al. (2012) iterative Kaplan-Meier algorithm
- Reconciles survival curve drops with Number at Risk table
- Deterministic event distribution within intervals
- Outputs CSV with columns: `time`, `event` (1=event, 0=censored), `group`

## Multi-Modality Processing Unit (MMPU)

The pipeline includes a GPT-4o vision-based module that can automatically extract structured metadata from any KM plot image:

- **Axis ranges** (min/max, units, tick values)
- **Legend labels** and curve colors
- **Number at Risk table** (timepoints and counts per group)
- **Number of curves**, censoring marks, confidence intervals

This eliminates the need for manual axis specification or OCR, matching the KM-GPT paper's architecture (Section 2.2).

```python
from pipeline.mmpu import analyze_km_plot

config = analyze_km_plot("km_plot.png")
# Returns: axis_ranges, n_curves, risk_table, group_labels
```

Requires an OpenAI API key set via `OPENAI_API_KEY` environment variable or `.env` file.

## Evaluation Results

Benchmarked against KM-GPT on 14 synthetic test cases covering edge cases (varying resolution, censoring rates, sample sizes, multi-arm trials, truncated axes, noise).

### Single-Seed Evaluation

| Metric | This Pipeline | KM-GPT Benchmark |
|---|---|---|
| Median IAE | **0.025** | 0.018 |
| 95th Percentile IAE | **0.068** | 0.088 |
| Success Rate | **100%** (14/14) | 99.6% |

All test cases fall within KM-GPT's 95% confidence interval (0.002-0.088).

### Powered Evaluation (Multi-Seed with Bootstrap CIs)

The powered evaluation runs each configuration across multiple random seeds (default: 50) to produce statistically robust benchmarks with 95% bootstrap confidence intervals. This addresses the need for adequately powered benchmarks rather than single-run point estimates.

**Curve Fidelity** = 1 - IAE, reported as a percentage. A curve fidelity of 97.5% means the reconstructed curve deviates by only 2.5% integrated area from the original.

```bash
python run_pipeline.py powered-evaluate --seeds 50
```

This generates:
- `powered_evaluation_report.txt`: Full statistical report with per-config CIs
- `powered_evaluation_results.json`: Machine-readable results
- `powered_evaluation_plots.png`: 4-panel figure (IAE histogram, per-config CIs, fidelity distribution, box plots)

Results from 5-seed powered evaluation (70 total runs, 100% success rate):

| Metric | This Pipeline | KM-GPT Benchmark |
|---|---|---|
| Mean IAE | **0.038** (95% CI: 0.032 - 0.045) | -- |
| Median IAE | **0.027** | 0.018 |
| Mean Curve Fidelity | **96.2%** (95% CI: 95.5% - 96.9%) | -- |
| Median Curve Fidelity | **97.3%** | -- |
| Runs within KM-GPT 95% CI | **94.3%** | -- |
| Success Rate | **100%** (70/70) | 99.6% |

## Installation

```bash
pip install -r requirements.txt
```

**Optional** (for MMPU - GPT-4o vision analysis):
```bash
pip install openai
```
Set your API key via environment variable or `.env` file:
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

**Optional** (for OCR-based axis extraction):
```bash
pip install pytesseract
# Also install Tesseract OCR engine: https://github.com/tesseract-ocr/tesseract
```

**Optional** (for improved multi-curve clustering):
```bash
pip install scikit-learn-extra
```

## Usage

### Command Line

The main entry point is `run_pipeline.py` with three commands:

#### Reconstruct IPD from a single image

```bash
python run_pipeline.py reconstruct \
    --image km_plot.png \
    --mask segmentation_mask.png \
    --n-curves 2 \
    --config config.json \
    --output ./output/results
```

Options:
- `--image` (required): Path to KM plot image
- `--mask`: Path to binary segmentation mask (auto-generated if not provided)
- `--n-curves`: Number of curves to extract (default: 1)
- `--config`: JSON config file with axis ranges and risk table
- `--preprocess`: Apply image preprocessing
- `--use-ocr`: Use OCR for axis/risk table extraction
- `--output`: Output directory (default: `./output`)

#### Run full evaluation on synthetic data

```bash
python run_pipeline.py evaluate --output ./output/evaluation
```

Generates 14 synthetic test cases, runs the pipeline on each, and produces:
- `evaluation_results.json`: Machine-readable metrics
- `evaluation_report.txt`: Detailed text report
- `evaluation_plots.png`: IAE/RMSE distribution plots
- `curve_comparison.png`: Original vs reconstructed curve overlays

#### Run demo

```bash
python run_pipeline.py demo --output ./output/demo
```

Generates a synthetic 2-arm trial, reconstructs IPD, and produces visualizations.

#### Analyze a KM plot with GPT-4o vision (MMPU)

```bash
python run_pipeline.py analyze --image km_plot.png --output ./output/analyze
python run_pipeline.py analyze --image km_plot.png --reconstruct --output ./output/analyze
```

Extracts axis ranges, legend, risk table, and curve count using GPT-4o. With `--reconstruct`, also runs the full IPD reconstruction pipeline using the extracted metadata.

#### Run powered evaluation with bootstrap CIs

```bash
python run_pipeline.py powered-evaluate --seeds 50 --output ./output/powered_evaluation
```

Runs each of 14 test configurations across N random seeds, producing statistically robust IAE and curve fidelity metrics with 95% bootstrap confidence intervals.

### Python API

```python
from pipeline import IPDRecoveryPipeline, RiskTable
import cv2

# Load image and mask
img = cv2.cvtColor(cv2.imread("km_plot.png"), cv2.COLOR_BGR2RGB)
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

# Configure pipeline
pipeline = IPDRecoveryPipeline(n_curves=1, use_preprocessing=True, use_ocr=False)

# Provide axis ranges and risk table
risk_table = RiskTable(
    timepoints=[0, 12, 24, 36, 48, 60],
    n_at_risk={"Group_1": [100, 85, 70, 55, 40, 25]},
    group_labels=["Group_1"]
)

result = pipeline.process(
    img=img,
    mask=mask,
    manual_axis_ranges={"x_range": (0, 60), "y_range": (0, 1.0)},
    manual_risk_table=risk_table
)

# Access results
for group, ipd in result["ipd"].items():
    print(f"{group}: {len(ipd)} patients, {sum(1 for r in ipd if r.event == 1)} events")

# Export to CSV
pipeline.export_ipd_to_csv(result["ipd"], "reconstructed_ipd.csv")
```

### Config File Format

```json
{
    "axis_ranges": {
        "x_range": [0, 60],
        "y_range": [0, 1.0]
    },
    "risk_table": {
        "timepoints": [0, 12, 24, 36, 48, 60],
        "n_at_risk": {
            "Treatment": [100, 85, 70, 55, 40, 25],
            "Control": [100, 80, 60, 45, 30, 15]
        },
        "group_labels": ["Treatment", "Control"]
    }
}
```

## Project Structure

```
KM_website/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── run_pipeline.py           # CLI entry point
├── pipeline/                    # Main package
│   ├── __init__.py              # Package exports
│   ├── core.py                  # IPD recovery pipeline (preprocessing, axis detection,
│   │                            #   curve extraction, KM reconstruction)
│   ├── evaluation.py            # Evaluation suite and benchmarking
│   ├── synthetic.py             # Synthetic KM plot generator with ground truth
│   ├── visualization.py         # Reconstruction visualization tools
│   ├── mmpu.py                  # Multi-Modality Processing Unit (GPT-4o vision)
│   └── powered_evaluation.py    # Powered evaluation with bootstrap CIs
└── output/                      # Generated outputs
    ├── evaluation/              # Single-seed evaluation results
    ├── powered_evaluation/      # Multi-seed powered evaluation results
    └── synthetic_data/          # Synthetic test images, masks, and ground truth IPD
```

## Key Modules

| Module | Description |
|---|---|
| `pipeline.core` | `IPDRecoveryPipeline` orchestrates the full pipeline. Contains `ImagePreprocessor`, `AdvancedAxisDetector`, `EnhancedCurveExtractor`, and `IPDReconstructor`. |
| `pipeline.evaluation` | `IPDEvaluator` benchmarks against KM-GPT using IAE, RMSE, median survival accuracy, and event count accuracy. |
| `pipeline.synthetic` | `SyntheticKMGenerator` creates KM plots with known ground truth. `SyntheticTestSuite` generates 14 edge-case test scenarios. |
| `pipeline.visualization` | `ReconstructionVisualizer` produces multi-panel comparison plots. |
| `pipeline.mmpu` | `MultiModalityProcessor` uses GPT-4o vision to extract structured metadata (axis ranges, legend, risk table) from KM plot images. |
| `pipeline.powered_evaluation` | Runs each test config across N random seeds with bootstrap 95% CIs for statistically powered benchmarking. |

## Test Cases

The evaluation suite includes 14 test scenarios:

| Test Case | Description | IAE |
|---|---|---|
| basic_single_arm | Standard 100-patient single arm | 0.025 |
| low_resolution | 400x300 @ 50 DPI | 0.076 |
| high_resolution | 1600x1200 @ 150 DPI | 0.021 |
| overlapping_curves | 3 groups with small hazard ratio differences | 0.035 |
| truncated_y_axis | Y-axis starts at 0.2 | 0.025 |
| heavy_censoring | 70% censoring rate | 0.064 |
| light_censoring | 5% censoring rate | 0.018 |
| small_sample | 30 patients | 0.050 |
| large_sample | 500 patients | 0.015 |
| short_followup | 12-month follow-up | 0.064 |
| long_followup | 120-month follow-up | 0.015 |
| noisy_image | Gaussian noise added | 0.025 |
| two_arm_trial | 2-arm RCT with HR=0.7 | 0.061 |
| no_risk_table | No risk table in image | 0.023 |

## References

- Guyot, P., Ades, A. E., Ouwens, M. J., & Welton, N. J. (2012). Enhanced secondary analysis of survival data: reconstructing the data from published Kaplan-Meier survival curves. *BMC Medical Research Methodology*, 12, 9.
- KM-GPT: A GPT-based approach for digitizing Kaplan-Meier curves. *arxiv:2509.18141v1*
