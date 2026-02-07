"""
Complete IPD Recovery Pipeline
Based on KM-GPT architecture (arxiv:2509.18141v1)

Pipeline Stages:
1. Vision Module: Extract curves and risk tables from KM plots
2. Data Alignment: Align temporal data from risk table with curve coordinates
3. IPD Generation: Reconstruct individual patient data using iterative algorithm
4. Validation: Compare reconstructed curves against ground truth

Author: Enhanced from km_curve_extractor.py
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False
    KMedoids = None
from sklearn.neighbors import KNeighborsClassifier
from scipy.interpolate import interp1d
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    pytesseract = None
from dataclasses import dataclass, field
import pandas as pd
import re


@dataclass
class AxisConfig:
    """Configuration for axis calibration"""
    x_start: float  # Real-world start value (e.g., 0 months)
    x_end: float    # Real-world end value (e.g., 60 months)
    y_start: float  # Real-world start value (typically 0)
    y_end: float    # Real-world end value (typically 1.0)
    x_start_px: int  # Pixel coordinate of x-axis start
    x_end_px: int    # Pixel coordinate of x-axis end
    y_start_px: int  # Pixel coordinate of y-axis start (bottom)
    y_end_px: int    # Pixel coordinate of y-axis end (top)


@dataclass
class RiskTable:
    """Risk table data extracted from KM plot"""
    timepoints: List[float]  # Time points where at-risk numbers are reported
    n_at_risk: Dict[str, List[int]]  # Number at risk per group
    group_labels: List[str] = field(default_factory=list)  # Group names
    
    def validate(self) -> bool:
        """Check if risk table is valid"""
        if not self.timepoints:
            return False
        for group, counts in self.n_at_risk.items():
            if len(counts) != len(self.timepoints):
                return False
        return True


@dataclass
class IPDRecord:
    """Individual patient data record"""
    time: float  # Survival time
    event: int   # Event indicator (1=event, 0=censored)
    group: str   # Treatment group


class ImagePreprocessor:
    """
    Image preprocessing module following KM-GPT Section 2.1
    Includes resolution enhancement, sharpening, and denoising
    """
    
    def __init__(self, enhance_resolution: bool = True, 
                 sharpen: bool = True, denoise: bool = True):
        self.enhance_resolution = enhance_resolution
        self.sharpen = sharpen
        self.denoise = denoise
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to KM plot image.
        Adapts processing based on image quality and resolution.

        Args:
            img: Input image (RGB or grayscale)

        Returns:
            Preprocessed image
        """
        # Convert to RGB if grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        h, w = img.shape[:2]
        is_high_res = w >= 1200

        # 1. Resolution enhancement - only for low-res images
        if self.enhance_resolution and not is_high_res:
            img = self._enhance_resolution(img)

        # 2. Sharpening - skip for high-res (can create ringing artifacts)
        if self.sharpen and not is_high_res:
            img = self._sharpen(img)

        # 3. Denoising - skip for high-res clean images
        if self.denoise and not is_high_res:
            img = self._denoise(img)

        return img
    
    def _enhance_resolution(self, img: np.ndarray, scale: int = 2) -> np.ndarray:
        """Enhance resolution using super-resolution, skip if already high-res"""
        h, w = img.shape[:2]
        # Skip upscaling if image is already high resolution (>1200px wide)
        if w >= 1200:
            return img
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    
    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        """Sharpen image using Laplacian kernel"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        return sharpened
    
    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising"""
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


class RiskTableExtractor:
    """
    Extract risk table (number at risk) from KM plot images
    Uses adaptive thresholding + OCR as per KM-GPT Section 2.2
    """
    
    def __init__(self, use_adaptive_threshold: bool = True):
        self.use_adaptive_threshold = use_adaptive_threshold
    
    def extract_risk_table(self, img: np.ndarray, 
                          bbox: Optional[Tuple[int, int, int, int]] = None) -> RiskTable:
        """
        Extract risk table from image
        
        Args:
            img: KM plot image (RGB)
            bbox: Optional bounding box (x, y, w, h) for risk table region
            
        Returns:
            RiskTable object with extracted data
        """
        # If bbox provided, crop to that region
        if bbox is not None:
            x, y, w, h = bbox
            img = img[y:y+h, x:x+w]
        else:
            # Try to auto-detect risk table region (bottom portion)
            h, w = img.shape[:2]
            img = img[int(h*0.7):, :]  # Bottom 30% typically contains risk table
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding for better OCR
        if self.use_adaptive_threshold:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR with configuration for structured data
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(binary, config=custom_config)
        
        # Parse OCR output
        risk_table = self._parse_risk_table_text(text)
        
        return risk_table
    
    def _parse_risk_table_text(self, text: str) -> RiskTable:
        """
        Parse OCR text output to extract risk table data
        
        Expected format:
        Number at risk
        Group A:  100  80  65  50  30
        Group B:   95  75  60  45  25
        
        Or:
        Time:      0   6  12  18  24
        Group A:  100  80  65  50  30
        """
        lines = text.strip().split('\n')
        lines = [l.strip() for l in lines if l.strip()]
        
        timepoints = []
        n_at_risk = {}
        group_labels = []
        
        # Pattern to match numbers
        number_pattern = r'\d+'
        
        for i, line in enumerate(lines):
            # Look for time row
            if any(keyword in line.lower() for keyword in ['time', 'month', 'year', 'day']):
                # Extract numbers from this line
                numbers = re.findall(number_pattern, line)
                timepoints = [float(n) for n in numbers if n.isdigit()]
                continue
            
            # Look for group rows (contain colon or group identifier)
            if ':' in line or any(keyword in line.lower() for keyword in ['group', 'arm', 'treatment']):
                # Split by colon if present
                parts = line.split(':') if ':' in line else ['Group', line]
                group_name = parts[0].strip() if len(parts) > 1 else f'Group_{len(group_labels)+1}'
                
                # Extract numbers
                numbers_str = parts[1] if len(parts) > 1 else line
                numbers = re.findall(number_pattern, numbers_str)
                counts = [int(n) for n in numbers]
                
                if counts:
                    group_labels.append(group_name)
                    n_at_risk[group_name] = counts
        
        # If no timepoints found, infer from number of counts
        if not timepoints and n_at_risk:
            # Use 0-based indexing as placeholder
            max_len = max(len(counts) for counts in n_at_risk.values())
            timepoints = list(range(max_len))
        
        # If no groups found, create single default group
        if not n_at_risk:
            # Try to extract any numbers as a single group
            all_numbers = []
            for line in lines:
                numbers = re.findall(number_pattern, line)
                all_numbers.extend([int(n) for n in numbers])
            
            if all_numbers:
                group_labels = ['Group_1']
                n_at_risk['Group_1'] = all_numbers[:len(timepoints)] if timepoints else all_numbers
        
        return RiskTable(
            timepoints=timepoints,
            n_at_risk=n_at_risk,
            group_labels=group_labels
        )


class AdvancedAxisDetector:
    """
    Enhanced axis detection with OCR integration
    Follows KM-GPT methodology for axis calibration
    """
    
    def __init__(self, use_ocr: bool = True):
        self.use_ocr = use_ocr
    
    def build_axis_config(self, img_rgb: np.ndarray, 
                         manual_ranges: Optional[Dict] = None) -> AxisConfig:
        """
        Build complete axis configuration from image
        
        Args:
            img_rgb: RGB image of KM plot
            manual_ranges: Optional dict with 'x_range' and 'y_range' tuples
            
        Returns:
            AxisConfig object
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Detect axis pixel boundaries
        x_axis_px, y_axis_px = self._detect_axis_boundaries(gray)
        
        # Extract tick labels and infer ranges
        if self.use_ocr and manual_ranges is None:
            ranges = self._extract_axis_ranges_ocr(img_rgb, x_axis_px, y_axis_px)
        else:
            ranges = manual_ranges or {'x_range': (0, 100), 'y_range': (0, 1.0)}
        
        return AxisConfig(
            x_start=ranges['x_range'][0],
            x_end=ranges['x_range'][1],
            y_start=ranges['y_range'][0],
            y_end=ranges['y_range'][1],
            x_start_px=x_axis_px[0],
            x_end_px=x_axis_px[1],
            y_start_px=y_axis_px[0],  # Bottom (higher y pixel value)
            y_end_px=y_axis_px[1]     # Top (lower y pixel value)
        )
    
    def _detect_axis_boundaries(self, gray: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Detect pixel boundaries of axes by finding actual axis lines.
        Uses horizontal/vertical line detection to find the x-axis and y-axis
        lines, which define the plot area boundaries.

        Returns:
            x_axis_px: (x_start, x_end) pixel positions
            y_axis_px: (y_start, y_end) pixel positions (start=bottom, end=top)
        """
        h, w = gray.shape

        # Create binary image of dark pixels (potential axis lines)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # --- Find x-axis (horizontal line): scan rows from bottom upward ---
        # The x-axis is a long horizontal run of dark pixels in the lower half.
        # Look for the row with the longest continuous horizontal run.
        best_x_row = None
        best_x_run_len = 0
        best_x_start = 0
        best_x_end = 0

        # Search in the lower 60% of image (x-axis is always below center)
        for row in range(h - 1, int(h * 0.4), -1):
            # Find runs of dark pixels in this row
            runs = self._find_runs(binary[row, :])
            for start, end in runs:
                run_len = end - start
                # x-axis should span at least 30% of image width
                if run_len > w * 0.3 and run_len > best_x_run_len:
                    best_x_run_len = run_len
                    best_x_row = row
                    best_x_start = start
                    best_x_end = end

        if best_x_row is not None:
            y_start_px = best_x_row  # bottom of plot = x-axis line
            x_start_px = best_x_start
            x_end_px = best_x_end
        else:
            # Fallback
            y_start_px = int(h * 0.8)
            x_start_px = int(w * 0.1)
            x_end_px = int(w * 0.9)

        # --- Find y-axis (vertical line): scan columns from left ---
        # The y-axis is a long vertical run of dark pixels in the left portion.
        best_y_col = None
        best_y_run_len = 0
        best_y_top = 0
        best_y_bottom = 0

        # Search in the left 40% of image
        for col in range(0, int(w * 0.4)):
            # Find runs of dark pixels in this column
            runs = self._find_runs(binary[:, col])
            for start, end in runs:
                run_len = end - start
                # y-axis should span at least 20% of image height
                if run_len > h * 0.2 and run_len > best_y_run_len:
                    best_y_run_len = run_len
                    best_y_col = col
                    best_y_top = start
                    best_y_bottom = end

        if best_y_col is not None:
            y_end_px = best_y_top  # top of plot
            # Refine y_start_px: use where y-axis meets x-axis
            y_start_px = min(y_start_px, best_y_bottom)
            x_start_px = best_y_col
        else:
            y_end_px = int(h * 0.1)

        return (x_start_px, x_end_px), (y_start_px, y_end_px)

    @staticmethod
    def _find_runs(arr: np.ndarray) -> List[Tuple[int, int]]:
        """Find runs of non-zero values in a 1D array."""
        runs = []
        in_run = False
        start = 0
        for i in range(len(arr)):
            if arr[i] > 0 and not in_run:
                in_run = True
                start = i
            elif arr[i] == 0 and in_run:
                in_run = False
                runs.append((start, i))
        if in_run:
            runs.append((start, len(arr)))
        return runs
    
    def _extract_axis_ranges_ocr(self, img_rgb: np.ndarray,
                                 x_axis_px: Tuple[int, int],
                                 y_axis_px: Tuple[int, int]) -> Dict:
        """Extract axis ranges using OCR on tick labels"""
        h, w = img_rgb.shape[:2]
        
        # Extract x-axis label region (bottom)
        x_region = img_rgb[y_axis_px[0]:, x_axis_px[0]:x_axis_px[1]]
        x_text = pytesseract.image_to_string(x_region, config='--psm 6')
        x_numbers = [float(n) for n in re.findall(r'\d+\.?\d*', x_text)]
        
        # Extract y-axis label region (left)
        y_region = img_rgb[y_axis_px[1]:y_axis_px[0], :x_axis_px[0]]
        y_text = pytesseract.image_to_string(y_region, config='--psm 6')
        y_numbers = [float(n) for n in re.findall(r'\d+\.?\d*', y_text)]
        
        # Infer ranges
        x_range = (min(x_numbers), max(x_numbers)) if x_numbers else (0, 100)
        y_range = (min(y_numbers), max(y_numbers)) if y_numbers else (0, 1.0)
        
        # Normalize y_range to [0, 1] if it appears to be percentage
        if y_range[1] > 10:
            y_range = (y_range[0]/100, y_range[1]/100)
        
        return {'x_range': x_range, 'y_range': y_range}


class EnhancedCurveExtractor:
    """
    Enhanced curve extraction using K-medoids clustering and k-NN classification
    Implements KM-GPT Section 2.3 methodology
    """
    
    def __init__(self, axis_config: AxisConfig, n_curves: int = 1):
        self.axis_config = axis_config
        self.n_curves = n_curves
    
    def extract_curves(self, mask: np.ndarray, img_rgb: np.ndarray) -> List[List[Tuple[float, float]]]:
        """
        Extract multiple curves from segmentation mask

        Args:
            mask: Binary mask where 255 = curve pixels
            img_rgb: Original RGB image (for color-based clustering)

        Returns:
            List of curves, each curve is list of (time, survival_prob) tuples
        """
        # Find all curve pixels
        curve_pixels = np.column_stack(np.where(mask > 128))

        if len(curve_pixels) == 0:
            return []

        # Filter pixels to plot area only (remove risk table text, axis labels, etc.)
        cfg = self.axis_config
        y_margin = max(3, int(0.01 * abs(cfg.y_start_px - cfg.y_end_px)))
        x_margin = max(3, int(0.01 * abs(cfg.x_end_px - cfg.x_start_px)))
        in_plot = (
            (curve_pixels[:, 0] >= cfg.y_end_px - y_margin) &
            (curve_pixels[:, 0] <= cfg.y_start_px + y_margin) &
            (curve_pixels[:, 1] >= cfg.x_start_px - x_margin) &
            (curve_pixels[:, 1] <= cfg.x_end_px + x_margin)
        )
        curve_pixels = curve_pixels[in_plot]

        if len(curve_pixels) == 0:
            return []

        # If single curve, process directly
        if self.n_curves == 1:
            curve = self._trace_single_curve(curve_pixels)
            return [curve]

        # Multiple curves: cluster by color
        curves = self._extract_multi_curves(curve_pixels, img_rgb)
        return curves
    
    def _extract_multi_curves(self, curve_pixels: np.ndarray, 
                              img_rgb: np.ndarray) -> List[List[Tuple[float, float]]]:
        """
        Extract multiple curves using K-medoids clustering in HSL color space
        """
        # Convert to HSL color space for better color discrimination
        img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
        
        # Extract color features for each curve pixel
        # OpenCV HLS: H in [0,180], L in [0,255], S in [0,255]
        # Hue is the primary discriminator between curve colors (blue vs orange vs green)
        # so we normalize all features to similar scales
        raw_features = []
        for y, x in curve_pixels:
            h, l, s = img_hls[y, x]
            raw_features.append([float(h), float(l), float(s)])
        raw_features = np.array(raw_features)

        # Normalize each channel to [0, 1] then weight hue more heavily
        # since hue distinguishes curve colors better than lightness/saturation
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        features = scaler.fit_transform(raw_features)
        # Weight hue 3x more than lightness and saturation
        features[:, 0] *= 3.0

        # KMeans clustering in normalized HLS color space
        kmeans = KMeans(n_clusters=self.n_curves, random_state=42, n_init=20)
        labels = kmeans.fit_predict(features)

        # Validate cluster balance - if any cluster has < 10% of pixels, retry
        # with spatial features to help break ties
        min_share = min(np.bincount(labels, minlength=self.n_curves)) / len(labels)
        if min_share < 0.10:
            y_norm = (curve_pixels[:, 0] - curve_pixels[:, 0].min()) / max(1, curve_pixels[:, 0].ptp())
            x_norm = (curve_pixels[:, 1] - curve_pixels[:, 1].min()) / max(1, curve_pixels[:, 1].ptp())
            spatial_weight = 0.3
            features_spatial = np.column_stack([features, y_norm * spatial_weight, x_norm * spatial_weight])
            kmeans2 = KMeans(n_clusters=self.n_curves, random_state=42, n_init=20)
            labels2 = kmeans2.fit_predict(features_spatial)
            min_share2 = min(np.bincount(labels2, minlength=self.n_curves)) / len(labels2)
            if min_share2 > min_share:
                labels = labels2
                features = features_spatial

        # Handle overlapping curves with k-NN (only if clusters are balanced)
        cluster_counts = np.bincount(labels, minlength=self.n_curves)
        max_ratio = max(cluster_counts) / max(1, min(cluster_counts))
        if max_ratio < 5.0:
            labels = self._refine_labels_knn(curve_pixels, features, labels)
        
        # Extract each curve separately
        curves = []
        for cluster_id in range(self.n_curves):
            cluster_pixels = curve_pixels[labels == cluster_id]
            if len(cluster_pixels) > 0:
                curve = self._trace_single_curve(cluster_pixels)
                curves.append(curve)
            else:
                curves.append([])
        
        return curves
    
    def _refine_labels_knn(self, pixels: np.ndarray, features: np.ndarray,
                          initial_labels: np.ndarray) -> np.ndarray:
        """
        Refine cluster labels using k-NN for overlapping regions.
        Uses combined color + spatial features for better reassignment.
        """
        # Build combined color+spatial features for k-NN
        y_norm = (pixels[:, 0] - pixels[:, 0].min()) / max(1, pixels[:, 0].ptp())
        x_norm = (pixels[:, 1] - pixels[:, 1].min()) / max(1, pixels[:, 1].ptp())
        spatial_weight = 10.0
        combined = np.column_stack([features, y_norm * spatial_weight, x_norm * spatial_weight])

        knn = KNeighborsClassifier(n_neighbors=min(7, len(pixels)))
        knn.fit(combined, initial_labels)

        probas = knn.predict_proba(combined)
        confidence = np.max(probas, axis=1)

        refined_labels = initial_labels.copy()
        low_conf_mask = confidence < 0.7

        if np.any(low_conf_mask):
            # Reassign low-confidence pixels using combined color+spatial distance
            high_conf_mask = ~low_conf_mask
            high_conf_combined = combined[high_conf_mask]
            high_conf_labels = initial_labels[high_conf_mask]

            if len(high_conf_combined) > 0:
                low_indices = np.where(low_conf_mask)[0]
                low_conf_combined = combined[low_conf_mask]
                # Batch nearest-neighbor lookup
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(high_conf_combined)
                _, indices = nn.kneighbors(low_conf_combined)
                refined_labels[low_indices] = high_conf_labels[indices.ravel()]

        return refined_labels
    
    def _trace_single_curve(self, curve_pixels: np.ndarray) -> List[Tuple[float, float]]:
        """
        Trace a single curve using median-y extraction per x-column.
        Using the median is robust to censoring marks (+) that extend
        above/below the curve line, unlike min-y which picks mark tops.
        """
        if len(curve_pixels) == 0:
            return []

        # Group pixels by x-column
        from collections import defaultdict
        x_to_ys = defaultdict(list)
        for y_px, x_px in curve_pixels:
            x_to_ys[int(x_px)].append(int(y_px))

        # For each column, use median y to robustly estimate curve position
        # Median is robust to censoring mark outliers above/below the line
        all_points = []
        for x_px in sorted(x_to_ys.keys()):
            y_values = x_to_ys[x_px]
            y_px = int(np.median(y_values))
            time, prob = self._pixel_to_real(x_px, y_px)
            all_points.append((time, prob))

        if not all_points:
            return []

        # Apply step structure enforcement
        path = self._enforce_step_structure(all_points)

        return path
    
    def _enforce_step_structure(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Enforce KM step function structure: horizontal segments with vertical drops.
        Uses adaptive thresholding based on the data range to detect genuine steps
        vs noise from line thickness.
        """
        if len(path) <= 1:
            return path

        # Sort by time
        path = sorted(path, key=lambda x: x[0])

        # Calculate adaptive threshold based on data resolution
        # A pixel-width noise in probability space
        cfg = self.axis_config
        if cfg.y_start_px - cfg.y_end_px != 0:
            prob_per_pixel = (cfg.y_end - cfg.y_start) / abs(cfg.y_start_px - cfg.y_end_px)
        else:
            prob_per_pixel = 0.005
        # Steps smaller than ~3 pixels of noise are likely artifacts
        prob_threshold = max(prob_per_pixel * 3, 0.005)

        # First pass: merge points at nearly the same probability into horizontal segments
        # For each segment, track the probability and the time range
        segments = []  # list of (t_start, t_end, prob)
        seg_start_t = path[0][0]
        seg_prob = path[0][1]

        for i in range(1, len(path)):
            curr_time, curr_prob = path[i]
            if abs(curr_prob - seg_prob) <= prob_threshold:
                # Same horizontal segment, extend it
                continue
            else:
                # End current segment, start new one
                segments.append((seg_start_t, path[i-1][0], seg_prob))
                seg_start_t = curr_time
                seg_prob = curr_prob
        # Add final segment
        segments.append((seg_start_t, path[-1][0], seg_prob))

        # Second pass: build step function from segments
        # Ensure monotone non-increasing (KM property)
        structured_path = []
        current_prob = segments[0][2]
        structured_path.append((segments[0][0], current_prob))

        for i in range(1, len(segments)):
            t_start, t_end, prob = segments[i]
            if prob < current_prob - prob_threshold:
                # Genuine step down - add horizontal extension then drop
                prev_t_end = segments[i-1][1]
                structured_path.append((prev_t_end, current_prob))  # extend horizontal
                structured_path.append((t_start, prob))  # step down
                current_prob = prob
            # If prob >= current_prob, ignore (enforce monotone non-increasing)

        # Add final point
        structured_path.append((segments[-1][1], current_prob))

        # Remove duplicate consecutive points
        cleaned = [structured_path[0]]
        for i in range(1, len(structured_path)):
            if structured_path[i] != cleaned[-1]:
                cleaned.append(structured_path[i])

        return cleaned
    
    def _pixel_to_real(self, x_px: int, y_px: int) -> Tuple[float, float]:
        """Convert pixel coordinates to real-world (time, probability)"""
        cfg = self.axis_config
        
        # X-axis (time)
        if cfg.x_end_px - cfg.x_start_px != 0:
            x_scale = (cfg.x_end - cfg.x_start) / (cfg.x_end_px - cfg.x_start_px)
            real_time = cfg.x_start + (x_px - cfg.x_start_px) * x_scale
        else:
            real_time = cfg.x_start
        
        # Y-axis (survival probability) - inverted because y increases downward
        if cfg.y_start_px - cfg.y_end_px != 0:
            pixels_from_top = y_px - cfg.y_end_px
            total_height_px = cfg.y_start_px - cfg.y_end_px
            fraction_from_top = pixels_from_top / total_height_px
            real_prob = cfg.y_end - fraction_from_top * (cfg.y_end - cfg.y_start)
        else:
            real_prob = cfg.y_end
        
        # Clamp to valid ranges
        real_time = np.clip(real_time, cfg.x_start, cfg.x_end)
        real_prob = np.clip(real_prob, cfg.y_start, cfg.y_end)
        
        return real_time, real_prob


class IPDReconstructor:
    """
    Reconstruct Individual Patient Data from digitized KM curves and risk tables
    Implements iterative Guyot's algorithm (iKM) as described in KM-GPT Section 2.3
    
    Reference: Guyot et al. (2012) BMC Medical Research Methodology
    """
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 100):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def reconstruct_ipd(self, curve: List[Tuple[float, float]], 
                       risk_table: RiskTable,
                       group_name: str = "Group_1") -> List[IPDRecord]:
        """
        Reconstruct IPD from digitized curve and risk table
        
        Args:
            curve: List of (time, survival_prob) tuples
            risk_table: RiskTable with number at risk information
            group_name: Name of the treatment group
            
        Returns:
            List of IPDRecord objects (individual patient data)
        """
        if not curve or not risk_table.validate():
            return []
        
        # Align curve with risk table timepoints
        aligned_curve = self._align_curve_to_risk_table(curve, risk_table)
        
        # Get number at risk for this group
        if group_name not in risk_table.n_at_risk:
            # Use first available group
            group_name = list(risk_table.n_at_risk.keys())[0]
        
        n_at_risk = risk_table.n_at_risk[group_name]
        
        # Iterative reconstruction using iKM algorithm
        ipd_records = self._ikm_algorithm(aligned_curve, risk_table.timepoints, 
                                         n_at_risk, group_name)
        
        # Validate and correct if needed
        ipd_records = self._validate_and_correct_ipd(ipd_records, risk_table, 
                                                     group_name, aligned_curve)
        
        return ipd_records
    
    def _validate_and_correct_ipd(self, ipd: List[IPDRecord], 
                                  risk_table: RiskTable,
                                  group_name: str,
                                  aligned_curve: List[Tuple[float, float]]) -> List[IPDRecord]:
        """
        Validate IPD against risk table and correct if needed
        """
        if not ipd:
            return ipd
        
        n_at_risk_expected = risk_table.n_at_risk[group_name]
        n_total_expected = n_at_risk_expected[0]
        
        # Check total patient count
        if len(ipd) != n_total_expected:
            # Adjust patient count
            if len(ipd) < n_total_expected:
                # Add censored patients at the end
                t_last = max(r.time for r in ipd) if ipd else risk_table.timepoints[-1]
                for _ in range(n_total_expected - len(ipd)):
                    ipd.append(IPDRecord(time=t_last, event=0, group=group_name))
            else:
                # Remove excess censored patients (keep events)
                events = [r for r in ipd if r.event == 1]
                censored = [r for r in ipd if r.event == 0]
                n_censored_to_keep = n_total_expected - len(events)
                ipd = events + censored[:n_censored_to_keep]
        
        return ipd
    
    def _align_curve_to_risk_table(self, curve: List[Tuple[float, float]], 
                                   risk_table: RiskTable) -> List[Tuple[float, float]]:
        """
        Interpolate survival probabilities at risk table timepoints
        Uses step function interpolation to preserve KM structure
        """
        if not curve:
            return []
        
        # Extract times and probabilities
        times = np.array([t for t, _ in curve])
        probs = np.array([p for _, p in curve])
        
        # Sort by time (should already be sorted, but make sure)
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        probs = probs[sort_idx]
        
        # Create interpolation function (piecewise constant for KM curves)
        # Use 'previous' to maintain step function property
        interp_func = interp1d(times, probs, kind='previous', 
                              bounds_error=False, fill_value=(probs[0], probs[-1]))
        
        # Interpolate at risk table timepoints
        aligned = []
        for t in risk_table.timepoints:
            # Make sure we don't go beyond the curve
            if t <= times[-1]:
                p = float(interp_func(t))
            else:
                # Extrapolate with last known probability
                p = float(probs[-1])
            aligned.append((t, p))
        
        return aligned
    
    def _ikm_algorithm(self, aligned_curve: List[Tuple[float, float]],
                      timepoints: List[float],
                      n_at_risk: List[int],
                      group_name: str) -> List[IPDRecord]:
        """
        Iterative Kaplan-Meier algorithm (Guyot et al. 2012)

        For each interval [t_{j-1}, t_j]:
          1. Start with n_{j-1} patients at risk
          2. Events d = n_{j-1} * (1 - S(t_j)/S(t_{j-1}))
          3. Censored c = n_{j-1} - d - n_j
          4. Distribute event/censoring times within the interval
        """
        ipd = []

        if not aligned_curve or not n_at_risk:
            return ipd

        n_total = n_at_risk[0]
        surv_at_times = [s for _, s in aligned_curve]

        all_events = []
        all_censored = []

        # Process each interval (start from interval 1)
        for i in range(1, len(timepoints)):
            t_curr = timepoints[i]
            t_prev = timepoints[i-1]

            s_curr = surv_at_times[i] if i < len(surv_at_times) else surv_at_times[-1]
            s_prev = surv_at_times[i-1]

            n_curr = n_at_risk[i]
            n_prev = n_at_risk[i-1]

            # Calculate events: d = n_prev * (1 - S(t_curr)/S(t_prev))
            if s_prev > 1e-6 and n_prev > 0:
                hazard = 1 - (s_curr / s_prev)
                hazard = max(0, min(1, hazard))
                n_events = int(np.round(n_prev * hazard))
                n_events = max(0, min(n_events, n_prev))
            else:
                n_events = 0

            # Censored = patients who left without events
            n_censored = n_prev - n_events - n_curr
            if n_censored < 0:
                # Adjust: more events than risk table accounts for
                n_events = max(0, n_prev - n_curr)
                n_censored = 0

            # Safety: total leaving cannot exceed those at risk
            if n_events + n_censored > n_prev:
                n_censored = max(0, n_prev - n_events)

            # Distribute events deterministically within interval
            interval_len = t_curr - t_prev
            for j in range(n_events):
                event_time = t_prev + interval_len * (j + 1) / (n_events + 1)
                all_events.append((event_time, 1))

            # Distribute censoring times within interval (after events)
            for j in range(n_censored):
                censor_time = t_prev + interval_len * (j + 0.5) / (n_censored + 1)
                all_censored.append((censor_time, 0))

        # Handle remaining patients at the final timepoint
        n_final = n_at_risk[-1]
        t_final = timepoints[-1]
        s_final = surv_at_times[-1] if surv_at_times else 0.0

        if s_final < 0.01:
            # Nearly all remaining patients had events
            for k in range(n_final):
                all_events.append((t_final, 1))
        else:
            # Remaining patients are administratively censored
            for k in range(n_final):
                all_censored.append((t_final, 0))

        # Create IPD records
        for time, event in all_events:
            ipd.append(IPDRecord(time=float(time), event=event, group=group_name))
        for time, event in all_censored:
            ipd.append(IPDRecord(time=float(time), event=event, group=group_name))

        # Ensure exact patient count
        current_total = len(ipd)
        if current_total < n_total:
            for _ in range(n_total - current_total):
                ipd.append(IPDRecord(time=float(t_final), event=0, group=group_name))
        elif current_total > n_total:
            events = [r for r in ipd if r.event == 1]
            censored = [r for r in ipd if r.event == 0]
            n_to_keep = n_total - len(events)
            ipd = events + censored[:max(0, n_to_keep)]

        ipd.sort(key=lambda x: x.time)
        return ipd
    
    def validate_reconstruction(self, ipd: List[IPDRecord], 
                               original_curve: List[Tuple[float, float]]) -> Dict:
        """
        Validate reconstructed IPD by computing KM estimate and comparing
        
        Returns:
            Dict with validation metrics (IAE, median survival, etc.)
        """
        if not ipd:
            return {'valid': False, 'iae': float('inf')}
        
        # Compute KM estimate from reconstructed IPD
        reconstructed_curve = self._compute_km_curve(ipd)
        
        # Calculate Integrated Absolute Error (IAE)
        iae = self._calculate_iae(original_curve, reconstructed_curve)
        
        # Calculate median survival
        median_original = self._calculate_median_survival(original_curve)
        median_reconstructed = self._calculate_median_survival(reconstructed_curve)
        
        return {
            'valid': True,
            'iae': iae,
            'median_original': median_original,
            'median_reconstructed': median_reconstructed,
            'median_ae': abs(median_original - median_reconstructed),
            'n_patients': len(ipd),
            'n_events': sum(1 for r in ipd if r.event == 1),
            'n_censored': sum(1 for r in ipd if r.event == 0)
        }
    
    def _compute_km_curve(self, ipd: List[IPDRecord]) -> List[Tuple[float, float]]:
        """Compute Kaplan-Meier curve from IPD"""
        if not ipd:
            return []
        
        # Sort by time
        sorted_ipd = sorted(ipd, key=lambda x: x.time)
        
        # Compute KM estimates
        n_total = len(sorted_ipd)
        n_at_risk = n_total
        survival_prob = 1.0
        
        km_curve = [(0.0, 1.0)]  # Start at time 0 with S=1
        
        i = 0
        while i < len(sorted_ipd):
            current_time = sorted_ipd[i].time
            
            # Count events at this exact time
            n_events = 0
            j = i
            while j < len(sorted_ipd) and sorted_ipd[j].time == current_time:
                if sorted_ipd[j].event == 1:
                    n_events += 1
                j += 1
            
            # Update survival probability
            if n_at_risk > 0 and n_events > 0:
                survival_prob *= (1 - n_events / n_at_risk)
                km_curve.append((current_time, survival_prob))
            
            # Update number at risk
            n_at_risk -= (j - i)
            i = j
        
        return km_curve
    
    def _calculate_iae(self, curve1: List[Tuple[float, float]], 
                      curve2: List[Tuple[float, float]]) -> float:
        """
        Calculate Integrated Absolute Error between two curves
        IAE = integral of |S1(t) - S2(t)| dt
        """
        if not curve1 or not curve2:
            return float('inf')
        
        # Get all unique timepoints
        times1 = np.array([t for t, _ in curve1])
        times2 = np.array([t for t, _ in curve2])
        all_times = np.unique(np.concatenate([times1, times2]))
        
        # Interpolate both curves
        probs1 = np.array([p for _, p in curve1])
        probs2 = np.array([p for _, p in curve2])
        
        interp1 = interp1d(times1, probs1, kind='previous', 
                          bounds_error=False, fill_value=(1.0, probs1[-1]))
        interp2 = interp1d(times2, probs2, kind='previous',
                          bounds_error=False, fill_value=(1.0, probs2[-1]))
        
        # Calculate IAE using trapezoidal rule
        s1_vals = interp1(all_times)
        s2_vals = interp2(all_times)
        
        abs_diff = np.abs(s1_vals - s2_vals)
        iae = np.trapz(abs_diff, all_times)
        
        # Normalize by time range
        time_range = all_times[-1] - all_times[0]
        if time_range > 0:
            iae = iae / time_range
        
        return float(iae)
    
    def _calculate_median_survival(self, curve: List[Tuple[float, float]]) -> float:
        """Calculate median survival time (time when S(t) = 0.5)"""
        for i in range(len(curve)):
            if curve[i][1] <= 0.5:
                if i == 0:
                    return curve[i][0]
                # Linear interpolation
                t0, s0 = curve[i-1]
                t1, s1 = curve[i]
                if s0 != s1:
                    median = t0 + (0.5 - s0) * (t1 - t0) / (s1 - s0)
                else:
                    median = t0
                return median
        
        # Median not reached
        return curve[-1][0] if curve else 0.0


class IPDRecoveryPipeline:
    """
    Complete end-to-end pipeline for IPD recovery from KM plots
    Integrates all components following KM-GPT architecture
    """
    
    def __init__(self, n_curves: int = 1, use_preprocessing: bool = True,
                 use_ocr: bool = True):
        """
        Args:
            n_curves: Expected number of curves in the plot
            use_preprocessing: Whether to apply image preprocessing
            use_ocr: Whether to use OCR for axis/risk table extraction
        """
        self.n_curves = n_curves
        self.use_preprocessing = use_preprocessing
        self.use_ocr = use_ocr
        
        # Initialize modules
        self.preprocessor = ImagePreprocessor() if use_preprocessing else None
        self.axis_detector = AdvancedAxisDetector(use_ocr=use_ocr)
        self.risk_extractor = RiskTableExtractor()
        self.ipd_reconstructor = IPDReconstructor()
    
    def process(self, img: np.ndarray, mask: np.ndarray,
                manual_axis_ranges: Optional[Dict] = None,
                risk_table_bbox: Optional[Tuple[int, int, int, int]] = None,
                manual_risk_table: Optional[RiskTable] = None) -> Dict:
        """
        Complete IPD recovery pipeline
        
        Args:
            img: RGB image of KM plot
            mask: Binary segmentation mask (255 = curve)
            manual_axis_ranges: Optional manual axis ranges
            risk_table_bbox: Optional bounding box for risk table region
            manual_risk_table: Optional manually provided risk table
            
        Returns:
            Dict containing:
                - ipd: List of IPDRecord objects per group
                - curves: Extracted curves
                - validation: Validation metrics
                - axis_config: Axis configuration used
        """
        # Stage 1: Preprocessing
        original_shape = img.shape[:2]
        if self.preprocessor and self.use_preprocessing:
            img = self.preprocessor.preprocess(img)
            # Resize mask to match preprocessed image dimensions
            new_shape = img.shape[:2]
            if new_shape != original_shape:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        # Stage 2: Axis detection and calibration
        axis_config = self.axis_detector.build_axis_config(
            img, manual_ranges=manual_axis_ranges
        )

        # Stage 3: Curve extraction
        curve_extractor = EnhancedCurveExtractor(axis_config, n_curves=self.n_curves)
        curves = curve_extractor.extract_curves(mask, img)
        
        # Stage 4: Risk table extraction
        if manual_risk_table is not None:
            risk_table = manual_risk_table
        elif self.use_ocr:
            risk_table = self.risk_extractor.extract_risk_table(img, risk_table_bbox)
        else:
            # Create dummy risk table for all curves
            risk_table = self._create_dummy_risk_table_multi(curves)
        
        # Stage 5: IPD reconstruction
        ipd_by_group = {}
        validation_by_group = {}
        
        for i, curve in enumerate(curves):
            if not curve:
                continue
            
            group_name = risk_table.group_labels[i] if i < len(risk_table.group_labels) \
                        else f"Group_{i+1}"
            
            # Reconstruct IPD
            ipd = self.ipd_reconstructor.reconstruct_ipd(curve, risk_table, group_name)
            ipd_by_group[group_name] = ipd
            
            # Validate reconstruction
            validation = self.ipd_reconstructor.validate_reconstruction(ipd, curve)
            validation_by_group[group_name] = validation
        
        return {
            'ipd': ipd_by_group,
            'curves': curves,
            'risk_table': risk_table,
            'axis_config': axis_config,
            'validation': validation_by_group
        }
    
    def _create_dummy_risk_table_multi(self, curves: List[List[Tuple[float, float]]]) -> RiskTable:
        """Create a dummy risk table for all curves when OCR is not available."""
        if not curves or not curves[0]:
            return RiskTable(timepoints=[], n_at_risk={}, group_labels=[])

        # Use the first curve's timepoints as the reference
        ref_curve = curves[0]
        all_times = sorted(set(t for t, _ in ref_curve))

        # Subsample to ~10 timepoints for a reasonable risk table
        if len(all_times) > 10:
            step = max(1, len(all_times) // 10)
            timepoints = [all_times[0]] + all_times[step::step]
            if timepoints[-1] != all_times[-1]:
                timepoints.append(all_times[-1])
        else:
            timepoints = all_times

        n_initial = 100
        n_at_risk = {}
        group_labels = []

        for i, curve in enumerate(curves):
            if not curve:
                continue
            group_name = f"Group_{i+1}"
            group_labels.append(group_name)

            # Interpolate survival at timepoints for this curve
            curve_times = np.array([t for t, _ in curve])
            curve_probs = np.array([p for _, p in curve])
            sort_idx = np.argsort(curve_times)
            curve_times = curve_times[sort_idx]
            curve_probs = curve_probs[sort_idx]

            interp_func = interp1d(curve_times, curve_probs, kind='previous',
                                   bounds_error=False, fill_value=(curve_probs[0], curve_probs[-1]))

            counts = []
            for t in timepoints:
                s = float(interp_func(t))
                counts.append(max(1, int(round(n_initial * s))))
            n_at_risk[group_name] = counts

        return RiskTable(
            timepoints=timepoints,
            n_at_risk=n_at_risk,
            group_labels=group_labels
        )
    
    def export_ipd_to_csv(self, ipd_by_group: Dict[str, List[IPDRecord]], 
                         filepath: str):
        """Export reconstructed IPD to CSV file"""
        records = []
        for group, ipd_list in ipd_by_group.items():
            for record in ipd_list:
                records.append({
                    'time': record.time,
                    'event': record.event,
                    'group': record.group
                })
        
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)
        print(f"IPD exported to {filepath}")
    
    def visualize_results(self, img: np.ndarray, result: Dict, 
                         save_path: Optional[str] = None):
        """
        Visualize extraction results with reconstructed curves overlaid
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Original image with extracted curves
        axes[0].imshow(img)
        axes[0].set_title('Extracted Curves')
        axes[0].axis('off')
        
        cfg = result['axis_config']
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, curve in enumerate(result['curves']):
            if not curve:
                continue
            
            # Convert to pixel coordinates
            xs, ys = [], []
            for time, prob in curve:
                x_px = cfg.x_start_px + (time - cfg.x_start) / (cfg.x_end - cfg.x_start) * \
                       (cfg.x_end_px - cfg.x_start_px)
                y_px = cfg.y_end_px + (cfg.y_end - prob) / (cfg.y_end - cfg.y_start) * \
                       (cfg.y_start_px - cfg.y_end_px)
                xs.append(x_px)
                ys.append(y_px)
            
            axes[0].plot(xs, ys, color=colors[i % len(colors)], 
                        linewidth=2, label=f'Curve {i+1}', alpha=0.7)
        
        axes[0].legend()
        
        # Right: Reconstructed survival curves
        axes[1].set_title('Validation: Original vs Reconstructed')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Survival Probability')
        axes[1].grid(True, alpha=0.3)
        
        for i, curve in enumerate(result['curves']):
            if not curve:
                continue
            
            group_name = list(result['ipd'].keys())[i] if i < len(result['ipd']) else f"Group_{i+1}"
            
            # Original curve
            times_orig = [t for t, _ in curve]
            probs_orig = [p for _, p in curve]
            axes[1].plot(times_orig, probs_orig, color=colors[i % len(colors)], 
                        linewidth=2, linestyle='-', label=f'{group_name} (original)', alpha=0.7)
            
            # Reconstructed curve from IPD
            if group_name in result['ipd']:
                ipd = result['ipd'][group_name]
                recon_curve = self.ipd_reconstructor._compute_km_curve(ipd)
                times_recon = [t for t, _ in recon_curve]
                probs_recon = [p for _, p in recon_curve]
                axes[1].plot(times_recon, probs_recon, color=colors[i % len(colors)], 
                            linewidth=2, linestyle='--', label=f'{group_name} (reconstructed)', 
                            alpha=0.7)
        
        axes[1].legend()
        axes[1].set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    print("IPD Recovery Pipeline - Based on KM-GPT")
    print("=" * 70)
    
    # This is a placeholder - actual usage would load real KM plot images
    # For now, demonstrate the API
    
    # Create synthetic example
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    mask = np.zeros((600, 800), dtype=np.uint8)
    
    # Draw simple curve
    times = np.linspace(100, 700, 50)
    probs = np.exp(-times / 400) * 0.9
    y_coords = (100 + (1 - probs) * 400).astype(int)
    
    for i in range(len(times)-1):
        cv2.line(mask, (int(times[i]), y_coords[i]), 
                (int(times[i+1]), y_coords[i+1]), 255, 2)
    
    # Initialize pipeline
    pipeline = IPDRecoveryPipeline(n_curves=1, use_preprocessing=False, use_ocr=False)
    
    # Manual configuration (in real scenario, could use OCR)
    manual_ranges = {'x_range': (0, 60), 'y_range': (0, 1.0)}
    manual_risk_table = RiskTable(
        timepoints=[0, 12, 24, 36, 48, 60],
        n_at_risk={'Group_1': [100, 85, 70, 55, 40, 25]},
        group_labels=['Group_1']
    )
    
    # Run pipeline
    result = pipeline.process(
        img, mask,
        manual_axis_ranges=manual_ranges,
        manual_risk_table=manual_risk_table
    )
    
    print("\nPipeline Results:")
    print(f"  Curves extracted: {len(result['curves'])}")
    
    for group, ipd in result['ipd'].items():
        print(f"\n  {group}:")
        print(f"    Total patients: {len(ipd)}")
        print(f"    Events: {sum(1 for r in ipd if r.event == 1)}")
        print(f"    Censored: {sum(1 for r in ipd if r.event == 0)}")
        
        if group in result['validation']:
            val = result['validation'][group]
            print(f"    IAE: {val['iae']:.6f}")
            print(f"    Median survival (original): {val['median_original']:.2f}")
            print(f"    Median survival (reconstructed): {val['median_reconstructed']:.2f}")
    
    # Export IPD
    pipeline.export_ipd_to_csv(result['ipd'], 'reconstructed_ipd.csv')
    
    print("\nPipeline demonstration complete!")