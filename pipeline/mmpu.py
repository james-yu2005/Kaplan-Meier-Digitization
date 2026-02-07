"""
Multi-Modality Processing Unit (MMPU)
Based on KM-GPT Section 2.2 (arxiv:2509.18141v1)

Uses GPT-4o vision to extract structured metadata from KM plot images:
- Axis ranges (x and y scale, units)
- Legend labels and curve colors
- Number of curves
- Risk table data (if present)
- Plot title and context

This module acts as the "brain" of the pipeline, resolving ambiguities
that pure OCR cannot handle (e.g., matching curve colors to legend labels,
interpreting axis units).
"""

import base64
import json
import os
from typing import Dict, Optional
from pathlib import Path

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None


# Load .env file if present
def _load_dotenv():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

_load_dotenv()


EXTRACTION_PROMPT = """You are an expert biostatistician analyzing a Kaplan-Meier survival curve image.

Extract the following information from this KM plot and return it as a JSON object:

{
  "title": "Plot title if visible, otherwise null",
  "n_curves": <integer: number of distinct survival curves>,
  "x_axis": {
    "label": "axis label (e.g., 'Time (months)')",
    "unit": "time unit (months, days, years, weeks)",
    "min": <number: minimum value on x-axis>,
    "max": <number: maximum value on x-axis>,
    "tick_values": [<list of visible tick mark values>]
  },
  "y_axis": {
    "label": "axis label (e.g., 'Survival Probability')",
    "min": <number: minimum value on y-axis, typically 0 or 0.0>,
    "max": <number: maximum value on y-axis, typically 1.0 or 100>,
    "is_percentage": <boolean: true if y-axis shows percentages (0-100), false if proportions (0-1)>,
    "tick_values": [<list of visible tick mark values>]
  },
  "legend": {
    "groups": [
      {
        "label": "group name from legend",
        "color": "approximate color (e.g., 'blue', 'red', 'orange')",
        "line_style": "solid, dashed, or dotted"
      }
    ]
  },
  "risk_table": {
    "present": <boolean: true if a 'Number at Risk' table is visible below the plot>,
    "timepoints": [<list of timepoint values from the table header>],
    "groups": [
      {
        "label": "group name",
        "counts": [<list of at-risk counts corresponding to timepoints>]
      }
    ]
  },
  "censoring_marks": <boolean: true if censoring tick marks (+) are visible on curves>,
  "confidence_intervals": <boolean: true if shaded confidence interval bands are visible>,
  "median_survival_lines": <boolean: true if dashed lines indicating median survival are drawn>
}

Rules:
- If a value is not visible or cannot be determined, use null
- For y-axis: determine if values represent proportions (0.0 to 1.0) or percentages (0 to 100)
- For risk table: read each number carefully, they indicate patients still at risk
- Match legend labels to curve colors precisely
- Count curves by distinct colored/styled lines, not by legend entries

Return ONLY the JSON object, no other text."""


class MultiModalityProcessor:
    """
    Multi-Modality Processing Unit (MMPU) for KM plot analysis.
    Uses GPT-4o vision API to extract structured plot metadata.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: OpenAI model to use (default: gpt-4o)
        """
        if not HAS_OPENAI:
            raise ImportError(
                "openai package is required for MMPU. Install with: pip install openai"
            )

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze a KM plot image and extract structured metadata.

        Args:
            image_path: Path to the KM plot image file

        Returns:
            Dict with extracted metadata (axis ranges, legend, risk table, etc.)
        """
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Determine MIME type
        ext = Path(image_path).suffix.lower()
        mime_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif', '.webp': 'image/webp'}
        mime_type = mime_map.get(ext, 'image/png')

        # Call GPT-4o vision
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.0,
        )

        # Parse response
        raw_text = response.choices[0].message.content.strip()
        metadata = self._parse_response(raw_text)

        # Add raw response for debugging
        metadata['_raw_response'] = raw_text
        metadata['_model'] = self.model

        return metadata

    def _parse_response(self, raw_text: str) -> Dict:
        """Parse GPT response into structured dict."""
        # Strip markdown code fences if present
        text = raw_text
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"_parse_error": True, "_raw_text": raw_text}

    def extract_pipeline_config(self, metadata: Dict) -> Dict:
        """
        Convert MMPU metadata into pipeline configuration.

        Args:
            metadata: Output from analyze_image()

        Returns:
            Dict with 'axis_ranges', 'n_curves', and optionally 'risk_table'
            ready to be passed to IPDRecoveryPipeline.process()
        """
        config = {}

        # Axis ranges
        x_axis = metadata.get('x_axis', {})
        y_axis = metadata.get('y_axis', {})

        x_min = x_axis.get('min', 0)
        x_max = x_axis.get('max', 100)
        y_min = y_axis.get('min', 0)
        y_max = y_axis.get('max', 1.0)

        # Convert percentages to proportions
        if y_axis.get('is_percentage', False):
            y_min = (y_min or 0) / 100.0
            y_max = (y_max or 100) / 100.0

        config['axis_ranges'] = {
            'x_range': (float(x_min or 0), float(x_max or 100)),
            'y_range': (float(y_min or 0), float(y_max or 1.0))
        }

        # Number of curves
        config['n_curves'] = metadata.get('n_curves', 1)

        # Risk table
        rt = metadata.get('risk_table', {})
        if rt.get('present') and rt.get('timepoints') and rt.get('groups'):
            from pipeline.core import RiskTable

            timepoints = [float(t) for t in rt['timepoints'] if t is not None]
            n_at_risk = {}
            group_labels = []

            for i, group in enumerate(rt['groups']):
                label = group.get('label', f'Group_{i+1}')
                counts = group.get('counts', [])
                # Ensure counts are integers and match timepoints length
                counts = [int(c) for c in counts if c is not None][:len(timepoints)]
                if len(counts) == len(timepoints):
                    n_at_risk[label] = counts
                    group_labels.append(label)

            if n_at_risk:
                config['risk_table'] = RiskTable(
                    timepoints=timepoints,
                    n_at_risk=n_at_risk,
                    group_labels=group_labels
                )

        # Legend info
        legend = metadata.get('legend', {})
        if legend.get('groups'):
            config['group_labels'] = [g.get('label', f'Group_{i+1}')
                                      for i, g in enumerate(legend['groups'])]

        return config


def analyze_km_plot(image_path: str, api_key: Optional[str] = None,
                    verbose: bool = True) -> Dict:
    """
    Convenience function to analyze a KM plot image.

    Args:
        image_path: Path to KM plot image
        api_key: OpenAI API key (optional, uses env var if not provided)
        verbose: Print extracted metadata

    Returns:
        Dict with pipeline configuration
    """
    mmpu = MultiModalityProcessor(api_key=api_key)

    if verbose:
        print(f"Analyzing image: {image_path}")
        print(f"Using model: {mmpu.model}")

    metadata = mmpu.analyze_image(image_path)

    if verbose:
        print("\nExtracted metadata:")
        # Print without raw response for readability
        display = {k: v for k, v in metadata.items() if not k.startswith('_')}
        print(json.dumps(display, indent=2))

    config = mmpu.extract_pipeline_config(metadata)

    if verbose:
        print(f"\nPipeline configuration:")
        print(f"  Axis ranges: x={config['axis_ranges']['x_range']}, "
              f"y={config['axis_ranges']['y_range']}")
        print(f"  Number of curves: {config['n_curves']}")
        if 'risk_table' in config:
            rt = config['risk_table']
            print(f"  Risk table: {len(rt.timepoints)} timepoints, "
                  f"{len(rt.group_labels)} groups")
        else:
            print("  Risk table: not detected")

    return config


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.mmpu <image_path>")
        print("\nAnalyzes a KM plot image using GPT-4o vision and extracts")
        print("axis ranges, legend info, risk table data, and more.")
        sys.exit(1)

    config = analyze_km_plot(sys.argv[1])
