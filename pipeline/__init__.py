"""
KM Curve Digitization Pipeline
Based on KM-GPT architecture (arxiv:2509.18141v1)

Modules:
    core              - IPD recovery pipeline (preprocessing, axis detection, curve extraction, reconstruction)
    evaluation        - Benchmarking and evaluation against KM-GPT
    synthetic         - Synthetic KM plot generation with ground truth
    visualization     - Reconstruction visualization and comparison plots
    mmpu              - Multi-Modality Processing Unit (GPT-4o vision for metadata extraction)
    powered_evaluation - Statistically powered evaluation with bootstrap CIs
"""

from pipeline.core import (
    IPDRecoveryPipeline,
    IPDRecord,
    RiskTable,
    AxisConfig,
    ImagePreprocessor,
    AdvancedAxisDetector,
    EnhancedCurveExtractor,
    IPDReconstructor,
)
from pipeline.synthetic import (
    SyntheticKMGenerator,
    SyntheticTestSuite,
    SyntheticConfig,
)
from pipeline.evaluation import IPDEvaluator
from pipeline.visualization import ReconstructionVisualizer

# Optional imports (require openai package)
try:
    from pipeline.mmpu import MultiModalityProcessor, analyze_km_plot
    HAS_MMPU = True
except ImportError:
    HAS_MMPU = False

__all__ = [
    "IPDRecoveryPipeline",
    "IPDRecord",
    "RiskTable",
    "AxisConfig",
    "ImagePreprocessor",
    "AdvancedAxisDetector",
    "EnhancedCurveExtractor",
    "IPDReconstructor",
    "SyntheticKMGenerator",
    "SyntheticTestSuite",
    "SyntheticConfig",
    "IPDEvaluator",
    "ReconstructionVisualizer",
    "MultiModalityProcessor",
    "analyze_km_plot",
]
