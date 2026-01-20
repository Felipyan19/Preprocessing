# @service/preprocessing.py
import logging
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

# ============================================================
# ONLY FEATURE: AUTO-ROTATION (make table "upright")
# - Primary: Tesseract OSD (Orientation and Script Detection)
# - Fallback: Table-line structure score analysis
# - Optionally applies small deskew correction (few degrees)
# ============================================================

def _detect_orientation_tesseract(img: np.ndarray) -> Optional[int]:
    """
    Use Tesseract OSD to detect orientation.
    Returns rotation angle in degrees (0, 90, 180, 270) or None if detection fails.

    Tesseract OSD is production-ready and trained specifically for orientation detection.
    Much more robust than manual heuristics.
    """
    try:
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Run Tesseract OSD (Orientation and Script Detection)
        # --psm 0 = Orientation and script detection (OSD) only
        osd_result = pytesseract.image_to_osd(gray, config='--psm 0')

        # Parse the OSD output
        # Example output:
        # Page number: 0
        # Orientation in degrees: 180
        # Rotate: 180
        # Orientation confidence: 15.24
        # Script: Latin
        # Script confidence: 3.24

        lines = osd_result.strip().split('\n')
        rotation_angle = None
        confidence = 0.0

        for line in lines:
            if 'Rotate:' in line:
                rotation_angle = int(line.split(':')[1].strip())
            elif 'Orientation confidence:' in line:
                confidence = float(line.split(':')[1].strip())

        # Only trust if confidence is reasonable (> 1.0)
        if rotation_angle is not None and confidence > 1.0:
            logger.info(
                "Tesseract OSD detected rotation: %d° (confidence: %.2f)",
                rotation_angle, confidence
            )
            return rotation_angle
        else:
            logger.warning(
                "Tesseract OSD low confidence (%.2f) or no rotation detected. Falling back to manual algorithm.",
                confidence
            )
            return None

    except Exception as e:
        logger.warning("Tesseract OSD failed: %s. Falling back to manual algorithm.", str(e))
        return None


def _rotate_90s(img: np.ndarray, deg: int) -> np.ndarray:
    """Rotate by multiples of 90 degrees."""
    deg = deg % 360
    if deg == 0:
        return img
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("Rotation must be one of {0,90,180,270}.")


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img


def _safe_kernel_width(v: int, min_v: int = 10, max_v: int = 120) -> int:
    return int(max(min_v, min(max_v, v)))


def _table_structure_score(gray: np.ndarray) -> Dict[str, float]:
    """
    Score how 'table-like' and 'upright' this orientation looks.
    Improved algorithm that better detects rotated tables by:
      - Detecting text orientation using projection profiles with peak detection
      - Analyzing line structure more robustly
      - Using multiple heuristics to determine best orientation
      - Focusing on table-like structures (rows/columns)
      - NEW: Top-heavy detection (tables have more content at top)
      - NEW: Gradient direction analysis (text gradients invert when upside-down)
      - NEW: Vertical text density distribution analysis
    """
    h, w = gray.shape[:2]

    if h < 10 or w < 10:
        return {"total": 0.0, "horiz": 0.0, "vert": 0.0, "edges": 0.0, "ink": 0.0, "text_orientation": 0.0, "top_heavy": 0.0, "gradient_coherence": 0.0}

    # Normalize contrast slightly to stabilize edge detection
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Morphology to extract horizontal/vertical strokes
    hk = _safe_kernel_width(w // 25)
    vk = _safe_kernel_width(h // 25)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    horiz = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vert = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    horiz_score = float(np.sum(horiz > 0)) / float(h * w)
    vert_score = float(np.sum(vert > 0)) / float(h * w)
    edge_density = float(np.sum(edges > 0)) / float(h * w)

    # Text-ish density: binarize and measure ink
    _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_density = float(np.sum(bin_inv > 0)) / float(h * w)

    # IMPROVED: Text orientation detection using projection profiles with peak counting
    # Horizontal projection (sum of pixels per row) - detects horizontal text lines
    h_projection = np.sum(bin_inv, axis=1)
    # Vertical projection (sum of pixels per column) - detects vertical text lines
    v_projection = np.sum(bin_inv, axis=0)
    
    # Calculate variance in projections - text lines create peaks/valleys
    h_proj_variance = float(np.var(h_projection)) if len(h_projection) > 0 else 0.0
    v_proj_variance = float(np.var(v_projection)) if len(v_projection) > 0 else 0.0
    
    # Count peaks in projections (more peaks = more text lines = better orientation)
    # A peak is a local maximum above a threshold
    h_mean = float(np.mean(h_projection)) if len(h_projection) > 0 else 0.0
    v_mean = float(np.mean(v_projection)) if len(v_projection) > 0 else 0.0
    
    h_threshold = h_mean * 1.2  # Peaks must be 20% above mean
    v_threshold = v_mean * 1.2
    
    h_peaks = np.sum((h_projection[1:-1] > h_projection[:-2]) & 
                     (h_projection[1:-1] > h_projection[2:]) & 
                     (h_projection[1:-1] > h_threshold)) if len(h_projection) > 2 else 0
    v_peaks = np.sum((v_projection[1:-1] > v_projection[:-2]) & 
                     (v_projection[1:-1] > v_projection[2:]) & 
                     (v_projection[1:-1] > v_threshold)) if len(v_projection) > 2 else 0
    
    # Normalize by image dimensions
    h_proj_score = h_proj_variance / (h * h) if h > 0 else 0.0
    v_proj_score = v_proj_variance / (w * w) if w > 0 else 0.0
    
    # Peak-based score (more peaks = more text lines = better)
    h_peak_score = float(h_peaks) / float(h) if h > 0 else 0.0
    v_peak_score = float(v_peaks) / float(w) if w > 0 else 0.0
    
    # Combined text orientation score: variance + peaks
    h_text_score = h_proj_score * 0.7 + h_peak_score * 0.3
    v_text_score = v_proj_score * 0.7 + v_peak_score * 0.3
    
    # Determine if text is horizontal or vertical
    if h_text_score > v_text_score * 1.1:  # At least 10% better
        # Horizontal text (normal orientation)
        text_orientation_score = h_text_score / (v_text_score + 1e-6) if v_text_score > 0 else h_text_score * 10
        text_is_horizontal = True
    elif v_text_score > h_text_score * 1.1:
        # Vertical text (rotated 90°)
        text_orientation_score = v_text_score / (h_text_score + 1e-6) if h_text_score > 0 else v_text_score * 10
        text_is_horizontal = False
    else:
        # Ambiguous - use variance ratio
        if h_proj_variance > v_proj_variance:
            text_orientation_score = h_proj_variance / (v_proj_variance + 1e-6)
            text_is_horizontal = True
        else:
            text_orientation_score = v_proj_variance / (h_proj_variance + 1e-6)
            text_is_horizontal = False
    
    # Normalize text orientation score
    text_orientation_normalized = min(text_orientation_score / 10.0, 1.0) if text_orientation_score > 0 else 0.0

    # ============================================================
    # NEW HEURISTICS FOR BETTER 180° DETECTION
    # ============================================================

    # 1. TOP-HEAVY DETECTION (IMPROVED)
    # Tables typically have headers at top, but we need to exclude barcodes/dense graphics
    # Strategy: Use narrow bands at very top and very bottom (5% each) to avoid middle content

    top_band_height = max(int(h * 0.05), 10)  # Top 5% or at least 10 pixels
    bottom_band_height = max(int(h * 0.05), 10)  # Bottom 5%

    top_band = bin_inv[:top_band_height, :]
    bottom_band = bin_inv[-bottom_band_height:, :]

    # Also analyze middle section to detect barcodes (very high horizontal line density)
    middle_start = int(h * 0.3)
    middle_end = int(h * 0.7)
    middle_section = bin_inv[middle_start:middle_end, :]

    # Calculate densities
    top_density = float(np.sum(top_band > 0)) / float(top_band.size) if top_band.size > 0 else 0.0
    bottom_density = float(np.sum(bottom_band > 0)) / float(bottom_band.size) if bottom_band.size > 0 else 0.0
    middle_density = float(np.sum(middle_section > 0)) / float(middle_section.size) if middle_section.size > 0 else 0.0

    # Detect barcode-like patterns in top and bottom bands
    # Barcodes have VERY high horizontal line density (vertical bars) and are CONSISTENT
    def has_barcode_pattern(region):
        if region.size == 0:
            return False
        # Count transitions per row (black-white-black-white pattern)
        row_transitions = []
        for row in region[:min(len(region), 50)]:  # Sample first 50 rows
            if len(row) < 10:
                continue
            transitions = np.sum(np.abs(np.diff(row > 128))) # Count transitions
            row_transitions.append(transitions)

        if len(row_transitions) < 10:  # Need at least 10 rows
            return False

        avg_transitions = float(np.mean(row_transitions))
        std_transitions = float(np.std(row_transitions))

        # Barcodes have:
        # 1. VERY high transitions (>60 per row typically, not just 20)
        # 2. LOW variance (consistent pattern across rows)
        has_high_transitions = avg_transitions > 60
        has_low_variance = std_transitions < (avg_transitions * 0.3)  # Variance < 30% of mean

        return has_high_transitions and has_low_variance

    top_has_barcode = has_barcode_pattern(top_band)
    bottom_has_barcode = has_barcode_pattern(bottom_band)

    # Debug logging for barcode detection
    logger.debug(
        "Barcode detection - top_has_barcode=%s, bottom_has_barcode=%s, top_density=%.5f, bottom_density=%.5f",
        top_has_barcode, bottom_has_barcode, top_density, bottom_density
    )

    # If top has barcode, it's likely upside down (barcodes usually at bottom)
    # If bottom has barcode, it's likely correct orientation
    if top_has_barcode and not bottom_has_barcode:
        # Likely upside down - penalize this orientation
        top_heavy_score = -0.8  # Strong signal that this is wrong
    elif bottom_has_barcode and not top_has_barcode:
        # Likely correct - reward this orientation
        top_heavy_score = 0.8  # Strong signal that this is correct
    else:
        # No clear barcode signal, use SMARTER density comparison
        # Instead of just comparing densities, analyze the PATTERN

        # For nutrition tables, typically:
        # - Top 10% has brand/title (medium-high density)
        # - Next 10-30% has table content (high density, structured)
        # - Bottom 10% might have legal text or codes (low-medium density)

        # Split into finer bands
        band_10 = max(int(h * 0.10), 10)
        very_top = bin_inv[:band_10, :]  # First 10%
        very_bottom = bin_inv[-band_10:, :]  # Last 10%

        very_top_density = float(np.sum(very_top > 0)) / float(very_top.size) if very_top.size > 0 else 0.0
        very_bottom_density = float(np.sum(very_bottom > 0)) / float(very_bottom.size) if very_bottom.size > 0 else 0.0

        # Analyze row density variance in top vs bottom quarters
        top_quarter = bin_inv[:h//4, :]
        bottom_quarter = bin_inv[-h//4:, :]

        top_row_densities = np.sum(top_quarter, axis=1) / float(w) if w > 0 else np.zeros(len(top_quarter))
        bottom_row_densities = np.sum(bottom_quarter, axis=1) / float(w) if w > 0 else np.zeros(len(bottom_quarter))

        top_quarter_variance = float(np.var(top_row_densities))
        bottom_quarter_variance = float(np.var(bottom_row_densities))

        # Correct orientation typically has:
        # 1. Higher variance at top (title + table start = varied density)
        # 2. Lower variance at bottom (uniform legal text or empty space)
        variance_ratio = top_quarter_variance / (bottom_quarter_variance + 1e-6)

        # Combine density difference with variance ratio
        density_diff = very_top_density - very_bottom_density

        # Weighted combination
        if variance_ratio > 1.3:  # Top has significantly more structure
            top_heavy_score = 0.3 + (density_diff * 0.3)  # Bias toward this being correct
        elif variance_ratio < 0.7:  # Bottom has more structure (likely inverted)
            top_heavy_score = -0.3 + (density_diff * 0.3)  # Bias toward this being inverted
        else:
            # Ambiguous, use density only
            top_heavy_raw = (top_density - bottom_density) / (top_density + bottom_density + 1e-6)
            top_heavy_score = float(top_heavy_raw) * 0.4

    # Normalize to [0, 1] range where 1 = very top-heavy (correct orientation)
    top_heavy_normalized = max(0.0, min(1.0, (top_heavy_score + 1.0) / 2.0))

    # 2. GRADIENT DIRECTION ANALYSIS
    # Text has characteristic gradients that invert when upside-down
    # Use Sobel to detect vertical gradient direction
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude and direction
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    gradient_dir = np.arctan2(sobely, sobelx)  # Range: [-π, π]

    # Focus on strong gradients (likely from text edges)
    strong_gradients_mask = gradient_mag > np.percentile(gradient_mag, 75)

    if np.sum(strong_gradients_mask) > 100:
        # Analyze vertical component of gradients
        # Positive sobely = gradient points downward (dark-to-light going down)
        # Negative sobely = gradient points upward (dark-to-light going up)
        # For normal text (dark on light), top edge has negative gradient, bottom has positive
        # For inverted text, this flips

        strong_sobely = sobely[strong_gradients_mask]

        # Calculate asymmetry in vertical gradients
        # Normal text should have balanced but slightly top-biased gradients
        positive_grad_count = np.sum(strong_sobely > 0)
        negative_grad_count = np.sum(strong_sobely < 0)
        total_strong = len(strong_sobely)

        # For white text on dark background (like your table), the gradient pattern is different
        # We look for coherence/consistency rather than specific direction
        gradient_balance = abs(positive_grad_count - negative_grad_count) / float(total_strong)

        # Higher coherence = more consistent gradient direction = more likely correct orientation
        gradient_coherence = 1.0 - gradient_balance  # Range [0, 1], higher = better
    else:
        gradient_coherence = 0.5  # Neutral if not enough gradients

    # 3. VERTICAL TEXT DENSITY DISTRIBUTION
    # Analyze how text density changes from top to bottom
    # Tables typically have higher density at top (headers) then periodic rows
    row_densities = np.sum(bin_inv, axis=1) / float(w) if w > 0 else np.zeros(h)

    # Calculate variance in top half vs bottom half
    mid = h // 2
    top_half_variance = float(np.var(row_densities[:mid])) if mid > 0 else 0.0
    bottom_half_variance = float(np.var(row_densities[mid:])) if mid > 0 else 0.0

    # Tables often have more structured variance in top portion (header + first rows)
    # This is a weak signal but helps in tie-breaking
    variance_ratio = top_half_variance / (bottom_half_variance + 1e-6) if bottom_half_variance > 0 else 1.0
    variance_score = min(1.0, variance_ratio / 2.0)  # Normalize

    # Improved dominance: consider both line structure and text orientation
    line_dominance = horiz_score - (0.85 * vert_score)
    
    # When text is horizontal, prefer horizontal dominance
    # When text is vertical (rotated 90°), prefer vertical dominance (which becomes horizontal after rotation)
    if text_is_horizontal:
        # Normal case: horizontal text, prefer horizontal lines
        dominance = (line_dominance * 0.5) + (text_orientation_normalized * 0.5)
    else:
        # Rotated 90° case: vertical text, prefer vertical lines (which will be horizontal after rotation)
        inverted_line_dominance = vert_score - (0.85 * horiz_score)
        dominance = (inverted_line_dominance * 0.5) + (text_orientation_normalized * 0.5)

    # Final score: mix ALL features including new heuristics
    # Rebalanced weights to incorporate top-heavy and gradient detection
    base_score = (
        dominance * 0.35 +                      # Structure and text orientation (reduced from 0.50)
        edge_density * 0.15 +                   # Overall edge density (reduced from 0.20)
        min(ink_density, 0.25) * 0.10 +        # Text density (reduced from 0.15)
        text_orientation_normalized * 0.10      # Text orientation (reduced from 0.15)
    )

    # NEW: Add orientation-specific heuristics (30% total weight)
    orientation_score = (
        top_heavy_normalized * 0.20 +           # Top-heavy detection (STRONG signal for 180°)
        gradient_coherence * 0.05 +             # Gradient coherence (medium signal)
        variance_score * 0.05                   # Variance distribution (weak signal)
    )

    total = base_score + orientation_score

    # Bonus: If text orientation is very strong, boost the score
    if text_orientation_normalized > 0.3:
        total *= 1.15  # 15% boost (reduced from 25% to balance with new heuristics)

    # Bonus: If we have many peaks (clear text lines), boost the score
    if text_is_horizontal and h_peaks > 5:
        total *= 1.08  # 8% boost (reduced from 10%)
    elif not text_is_horizontal and v_peaks > 5:
        total *= 1.08  # 8% boost (reduced from 10%)

    # CRITICAL BONUS: If very top-heavy (clear header at top), boost significantly
    # This helps distinguish 0° from 180° even when other scores are similar
    if top_heavy_normalized > 0.65:  # Strong top-heavy signal
        total *= 1.35  # 35% boost for very clear top-heavy orientation

    return {
        "total": total,
        "horiz": horiz_score,
        "vert": vert_score,
        "edges": edge_density,
        "ink": ink_density,
        "text_orientation": text_orientation_normalized,
        "top_heavy": top_heavy_normalized,
        "gradient_coherence": gradient_coherence,
        "variance_score": variance_score,
        "barcode_detected_top": top_has_barcode,
        "barcode_detected_bottom": bottom_has_barcode,
    }


def _estimate_small_skew_angle(gray: np.ndarray, max_abs_deg: float = 10.0) -> float:
    """
    Estimate small skew (few degrees) based on dominant line angles.
    Returns angle in degrees to rotate (positive = counter-clockwise).
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(40, gray.shape[1] // 8),
        maxLineGap=10,
    )

    if lines is None or len(lines) < 5:
        return 0.0

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = (x2 - x1)
        dy = (y2 - y1)
        if dx == 0 and dy == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))  # [-180, 180]
        # Normalize so near-horizontal angles cluster around 0
        # e.g. 179 -> -1, -179 -> 1
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        # Keep only near-horizontal lines (most relevant for table rows)
        if abs(angle) <= 30:
            angles.append(angle)

    if len(angles) < 5:
        return 0.0

    # Robust central tendency
    angle_med = float(np.median(angles))
    if abs(angle_med) < 0.3:
        return 0.0
    if abs(angle_med) > max_abs_deg:
        return 0.0
    return angle_med


def _rotate_arbitrary(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image by an arbitrary small angle preserving full canvas."""
    if abs(angle_deg) < 1e-6:
        return img

    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]

    return cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def auto_rotate_table_upright(
    img: np.ndarray,
    *,
    enable_deskew: bool = True,
    deskew_max_abs_deg: float = 8.0,
    min_improvement: float = 0.0,  # Set to 0 for maximum aggressiveness - always rotate if best != 0°
    use_tesseract: bool = True,  # Try Tesseract OSD first
) -> Tuple[np.ndarray, Dict]:
    """
    Auto-rotate image so table becomes upright.

    Steps:
    1) Try Tesseract OSD (Orientation and Script Detection) first - most reliable
    2) If Tesseract fails, score 0/90/180/270 and pick best using manual algorithm
    3) If enabled, apply small deskew (few degrees) using Hough line angles

    Args:
        enable_deskew: whether to do small-angle correction after 90° rotation
        deskew_max_abs_deg: maximum absolute deskew degrees to apply
        min_improvement: if best orientation isn't better than original by this ratio, keep original
        use_tesseract: whether to try Tesseract OSD first (recommended)

    Returns:
        (rotated_img, metadata)
    """
    best_deg = 0
    meta: Dict = {
        "auto_rotation_applied": False,
        "rotation_degrees": 0,
        "scores": {},
        "improvement_ratio": 0.0,
        "deskew_applied": False,
        "deskew_angle_deg": 0.0,
        "method": "none",
    }

    # Step 1: Try Tesseract OSD first (production-ready, very reliable)
    if use_tesseract:
        tesseract_rotation = _detect_orientation_tesseract(img)
        if tesseract_rotation is not None:
            best_deg = tesseract_rotation
            meta["method"] = "tesseract_osd"
            meta["auto_rotation_applied"] = (best_deg != 0)
            meta["rotation_degrees"] = best_deg

            logger.info(
                "Using Tesseract OSD result: %d° rotation detected",
                best_deg
            )

            # Apply rotation and proceed to deskew
            out = _rotate_90s(img, best_deg)

            # Optional small-angle deskew after coarse rotation
            if enable_deskew:
                gray = _to_gray(out)
                angle = _estimate_small_skew_angle(gray, max_abs_deg=deskew_max_abs_deg)

                # We want to counter the skew => rotate by -angle
                if abs(angle) >= 0.3:
                    out = _rotate_arbitrary(out, -angle)
                    meta["deskew_applied"] = True
                    meta["deskew_angle_deg"] = float(angle)
                    logger.info("Deskew applied: %.2f° (rotated by %.2f°)", angle, -angle)

            logger.info("Auto-rotation result (Tesseract): %d° (deskew=%s)", meta["rotation_degrees"], meta["deskew_applied"])
            return out, meta

    # Step 2: Fallback to manual algorithm if Tesseract failed or disabled
    logger.info("Using manual orientation detection algorithm (fallback)")
    meta["method"] = "manual_heuristics"

    gray0 = _to_gray(img)

    candidates = {}
    for deg in (0, 90, 180, 270):
        rot = _rotate_90s(gray0, deg)
        candidates[deg] = _table_structure_score(rot)

    # Get scores for all orientations
    score_0 = candidates[0]["total"]
    score_90 = candidates[90]["total"]
    score_180 = candidates[180]["total"]
    score_270 = candidates[270]["total"]
    
    # Find best orientation
    best_deg = max(candidates, key=lambda d: candidates[d]["total"])
    best_score = candidates[best_deg]["total"]
    base_score = score_0
    
    # Log scores for debugging
    logger.info(
        "Rotation scores - 0°: %.5f, 90°: %.5f, 180°: %.5f, 270°: %.5f (best: %d°)",
        score_0, score_90, score_180, score_270, best_deg
    )
    
    # Calculate improvement
    improvement = 0.0
    if abs(base_score) > 1e-9:
        improvement = (best_score - base_score) / abs(base_score)
    else:
        improvement = 1.0 if best_deg != 0 else 0.0

    # Improved rotation logic: Always prefer 90°/270° when they have good scores
    should_rotate = False
    final_deg = best_deg
    
    # Improved logic: Check all possible scenarios and choose the best orientation
    # Tolerance for "ties" - scores very close to each other
    tolerance = max(best_score * 0.02, 0.001)  # 2% of best score or 0.001, whichever is larger
    
    # Find all orientations within tolerance of the best score
    close_scores = [d for d in [0, 90, 180, 270] if abs(candidates[d]["total"] - best_score) <= tolerance]
    
    # Log detailed scores for debugging (including NEW heuristics + barcode detection)
    logger.info(
        "Detailed scores - 0°: %.5f (h=%.5f, v=%.5f, txt=%.5f, top=%.5f, grad=%.5f, bc_t=%s, bc_b=%s), "
        "90°: %.5f (h=%.5f, v=%.5f, txt=%.5f, top=%.5f, grad=%.5f, bc_t=%s, bc_b=%s), "
        "180°: %.5f (h=%.5f, v=%.5f, txt=%.5f, top=%.5f, grad=%.5f, bc_t=%s, bc_b=%s), "
        "270°: %.5f (h=%.5f, v=%.5f, txt=%.5f, top=%.5f, grad=%.5f, bc_t=%s, bc_b=%s)",
        score_0, candidates[0].get("horiz", 0), candidates[0].get("vert", 0), candidates[0].get("text_orientation", 0),
        candidates[0].get("top_heavy", 0), candidates[0].get("gradient_coherence", 0),
        candidates[0].get("barcode_detected_top", False), candidates[0].get("barcode_detected_bottom", False),
        score_90, candidates[90].get("horiz", 0), candidates[90].get("vert", 0), candidates[90].get("text_orientation", 0),
        candidates[90].get("top_heavy", 0), candidates[90].get("gradient_coherence", 0),
        candidates[90].get("barcode_detected_top", False), candidates[90].get("barcode_detected_bottom", False),
        score_180, candidates[180].get("horiz", 0), candidates[180].get("vert", 0), candidates[180].get("text_orientation", 0),
        candidates[180].get("top_heavy", 0), candidates[180].get("gradient_coherence", 0),
        candidates[180].get("barcode_detected_top", False), candidates[180].get("barcode_detected_bottom", False),
        score_270, candidates[270].get("horiz", 0), candidates[270].get("vert", 0), candidates[270].get("text_orientation", 0),
        candidates[270].get("top_heavy", 0), candidates[270].get("gradient_coherence", 0),
        candidates[270].get("barcode_detected_top", False), candidates[270].get("barcode_detected_bottom", False)
    )
    
    # Decision logic: prioritize based on NEW heuristics when there are ties
    if len(close_scores) > 1:
        # Multiple orientations have similar scores - use orientation-specific heuristics to break tie
        # Priority: top_heavy > text_orientation > gradient_coherence

        best_top_heavy = -1.0
        best_by_top_heavy = 0

        # First, try to break tie using top_heavy score (most reliable for 0° vs 180°)
        for deg in close_scores:
            top_heavy = candidates[deg].get("top_heavy", 0.0)
            if top_heavy > best_top_heavy:
                best_top_heavy = top_heavy
                best_by_top_heavy = deg

        # Check if top_heavy clearly favors one orientation
        if best_top_heavy > 0.55:  # Strong top-heavy signal (above neutral 0.5)
            final_deg = best_by_top_heavy
            should_rotate = (best_by_top_heavy != 0)
            logger.info(
                "Auto-rotation: Multiple orientations close (tolerance=%.5f). Using top-heavy to choose %d° (top_heavy=%.5f).",
                tolerance, final_deg, best_top_heavy
            )
        else:
            # Top-heavy not decisive, try text orientation
            best_text_orient = -1.0
            best_by_text = 0

            for deg in close_scores:
                text_orient = candidates[deg].get("text_orientation", 0.0)
                if text_orient > best_text_orient:
                    best_text_orient = text_orient
                    best_by_text = deg

            # Check if text orientation clearly favors one orientation
            if best_text_orient > 0.1:  # At least some text orientation signal
                final_deg = best_by_text
                should_rotate = (best_by_text != 0)
                logger.info(
                    "Auto-rotation: Multiple orientations close (tolerance=%.5f). Using text orientation to choose %d° (txt_score=%.5f).",
                    tolerance, final_deg, best_text_orient
                )
            else:
                # Neither top-heavy nor text orientation helpful, use the best total score
                # But if best is not 0°, always rotate
                final_deg = best_deg
                should_rotate = (best_deg != 0)
                if should_rotate:
                    logger.info(
                        "Auto-rotation: Multiple orientations close. Best score is %d° (%.5f). Rotating.",
                        final_deg, best_score
                    )
    elif best_deg != 0:
        # No tie, best is clearly different from 0°
        # ALWAYS rotate if best is not 0° - be very aggressive
        # Case 1: Best is 90° or 270° - ALWAYS rotate
        if best_deg in [90, 270]:
            should_rotate = True
            logger.info(
                "Auto-rotation: 90°/270° orientation detected (%d°, score=%.5f vs 0°=%.5f). Rotating.",
                best_deg, best_score, base_score
            )
        # Case 2: Best is 180° - rotate if improvement is positive or scores are very close
        elif best_deg == 180:
            if improvement > 0 or abs(best_score - base_score) < tolerance:
                should_rotate = True
                logger.info(
                    "Auto-rotation: 180° orientation detected (score=%.5f vs 0°=%.5f). Rotating.",
                    best_score, base_score
                )
        # Case 3: Any improvement - rotate
        elif improvement > 0:
            should_rotate = True
            logger.info(
                "Auto-rotation: positive improvement detected (%d°, improve=%.2f%%). Rotating.",
                final_deg, improvement * 100.0
            )
        # Case 4: Scores are very close but best is not 0° - still rotate
        elif abs(best_score - base_score) < tolerance * 2:
            should_rotate = True
            logger.info(
                "Auto-rotation: Scores close but best is %d° (%.5f vs 0°=%.5f). Rotating.",
                best_deg, best_score, base_score
            )
    
    if not should_rotate:
        logger.info(
            "Auto-rotation: No rotation needed (best=%d score=%.5f vs base=%.5f, improve=%.1f%%). Keeping 0°.",
            best_deg, best_score, base_score, improvement * 100.0
        )
        final_deg = 0
    
    best_deg = final_deg

    out = _rotate_90s(img, best_deg)
    meta: Dict = {
        "auto_rotation_applied": best_deg != 0,
        "rotation_degrees": best_deg,
        "scores": {k: v for k, v in candidates.items()},
        "improvement_ratio": improvement,
        "deskew_applied": False,
        "deskew_angle_deg": 0.0,
    }

    # Optional small-angle deskew after coarse rotation
    if enable_deskew:
        gray = _to_gray(out)
        angle = _estimate_small_skew_angle(gray, max_abs_deg=deskew_max_abs_deg)

        # We want to counter the skew => rotate by -angle
        if abs(angle) >= 0.3:
            out = _rotate_arbitrary(out, -angle)
            meta["deskew_applied"] = True
            meta["deskew_angle_deg"] = float(angle)
            logger.info("Deskew applied: %.2f° (rotated by %.2f°)", angle, -angle)

    logger.info("Auto-rotation result: %d° (deskew=%s)", meta["rotation_degrees"], meta["deskew_applied"])
    return out, meta


def _enhance_table_structure(img: np.ndarray, enhance_edges: bool = True, enhance_contrast: bool = True, aggressive: bool = False) -> np.ndarray:
    """
    Enhance table structure for better detection by Azure Vision.
    
    Args:
        img: Input image
        enhance_edges: Enhance table borders/lines
        enhance_contrast: Improve contrast for better structure visibility
        aggressive: Use more aggressive edge enhancement (thicker lines)
    
    Returns:
        Enhanced image
    """
    # Convert to grayscale if needed (better for table detection)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    if enhance_contrast:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This improves contrast without over-amplifying noise
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    if enhance_edges:
        # More aggressive edge enhancement for table detection
        if aggressive:
            # Use larger kernel for more visible borders
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (gray.shape[1] // 20, 1))
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gray.shape[0] // 20))
            
            # Detect horizontal and vertical lines
            horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_h, iterations=2)
            vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_v, iterations=2)
            
            # Combine horizontal and vertical lines
            table_structure = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0.0)
            
            # Enhance the original image with detected structure
            gray = cv2.addWeighted(gray, 0.7, table_structure, 0.3, 0.0)
            
            # Apply additional sharpening to make borders crisp
            kernel_sharpen = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]])
            gray = cv2.filter2D(gray, -1, kernel_sharpen)
        else:
            # Lighter enhancement - detect and strengthen existing lines
            # Use Canny edge detection to find edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to make them thicker
            kernel = np.ones((2, 2), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Combine with original to strengthen borders
            gray = cv2.addWeighted(gray, 0.9, edges_dilated, 0.1, 0.0)
            
            # Light morphological operations
            kernel = np.ones((2, 2), np.uint8)
            gray = cv2.dilate(gray, kernel, iterations=1)
            gray = cv2.erode(gray, kernel, iterations=1)
    
    # Convert back to BGR if original was color
    if len(img.shape) == 3:
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        result = gray
    
    return result


def _detect_and_rotate_table_region(img: np.ndarray) -> np.ndarray:
    """
    Detecta regiones con estructura tabular rotadas y las rota para dejarlas horizontales.
    Busca regiones con texto alineado verticalmente (rotadas 90°).
    Usa análisis de proyección para detectar regiones con estructura tabular vertical.
    
    Args:
        img: Input image
    
    Returns:
        Image with detected table regions rotated
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    h, w = gray.shape[:2]
    result = img.copy()
    
    try:
        # Binarize image for better text detection
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Analyze vertical projection to find regions with vertical text structure
        # Vertical projection: sum of pixels per column
        vertical_projection = np.sum(binary, axis=0)
        
        # Find regions with high vertical density (potential vertical text columns)
        threshold = np.mean(vertical_projection) * 1.5
        vertical_mask = vertical_projection > threshold
        
        # Find contiguous regions
        vertical_regions = []
        in_region = False
        start = 0
        
        for i in range(len(vertical_mask)):
            if vertical_mask[i] and not in_region:
                start = i
                in_region = True
            elif not vertical_mask[i] and in_region:
                end = i
                if end - start > w * 0.05:  # At least 5% of image width
                    vertical_regions.append((start, end))
                in_region = False
        
        if in_region:
            end = len(vertical_mask)
            if end - start > w * 0.05:
                vertical_regions.append((start, end))
        
        # Now analyze horizontal projection for each vertical region
        for x_start, x_end in vertical_regions:
            region_width = x_end - x_start
            region_binary = binary[:, x_start:x_end]
            
            # Horizontal projection for this region
            horizontal_projection = np.sum(region_binary, axis=1)
            
            # Find vertical extent of text in this region
            h_threshold = np.mean(horizontal_projection) * 1.2
            h_mask = horizontal_projection > h_threshold
            
            # Find vertical bounds
            y_indices = np.where(h_mask)[0]
            if len(y_indices) > h * 0.1:  # At least 10% of image height
                y_start = max(0, y_indices[0] - 20)
                y_end = min(h, y_indices[-1] + 20)
                region_height = y_end - y_start
                
                # Check if this region has vertical text characteristics
                # (height > width suggests vertical text)
                if region_height > region_width * 1.2:
                    # Extract region
                    region = img[y_start:y_end, x_start:x_end].copy()
                    
                    # Rotate 90° counter-clockwise
                    rotated_region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    # Calculate new position after rotation
                    # After 90° CCW: (x, y, w, h) -> (x, y, h, w)
                    new_width = region_height
                    new_height = region_width
                    
                    # Center the rotated region
                    center_x = x_start + region_width // 2
                    center_y = y_start + region_height // 2
                    
                    new_x_start = max(0, center_x - new_width // 2)
                    new_y_start = max(0, center_y - new_height // 2)
                    new_x_end = min(w, new_x_start + new_width)
                    new_y_end = min(h, new_y_start + new_height)
                    
                    # Adjust if out of bounds
                    if new_x_end > w:
                        new_x_start = w - new_width
                    if new_y_end > h:
                        new_y_start = h - new_height
                    
                    # Resize rotated region if needed
                    if rotated_region.shape[1] != (new_x_end - new_x_start) or rotated_region.shape[0] != (new_y_end - new_y_start):
                        rotated_region = cv2.resize(rotated_region, (new_x_end - new_x_start, new_y_end - new_y_start), interpolation=cv2.INTER_CUBIC)
                    
                    # Place rotated region
                    if new_x_start >= 0 and new_y_start >= 0 and new_x_end <= w and new_y_end <= h:
                        result[new_y_start:new_y_end, new_x_start:new_x_end] = rotated_region
                        logger.info("Región de tabla rotada detectada y corregida: (%d,%d)-(%d,%d) -> (%d,%d)-(%d,%d)", 
                                  x_start, y_start, x_end, y_end, new_x_start, new_y_start, new_x_end, new_y_end)
    
    except Exception as e:
        logger.warning("Error al detectar y rotar región de tabla: %s", e)
    
    return result


def _draw_table_borders(img: np.ndarray) -> np.ndarray:
    """
    Detecta la estructura de una tabla y dibuja bordes explícitos para que Azure Vision la reconozca.
    Usa análisis de proyecciones para detectar filas y columnas, luego dibuja líneas.
    
    Args:
        img: Input image
    
    Returns:
        Image with explicit table borders drawn
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    h, w = gray.shape[:2]
    result = img.copy()
    
    try:
        # Binarize image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection: detect rows
        h_projection = np.sum(binary, axis=1)
        h_mean = np.mean(h_projection)
        h_threshold = h_mean * 0.3  # Lower threshold to catch more rows
        
        # Find row boundaries (peaks in horizontal projection)
        row_boundaries = []
        in_row = False
        row_start = 0
        
        for i in range(len(h_projection)):
            if h_projection[i] > h_threshold and not in_row:
                row_start = i
                in_row = True
            elif h_projection[i] <= h_threshold and in_row:
                row_end = i
                row_center = (row_start + row_end) // 2
                row_boundaries.append(row_center)
                in_row = False
        
        if in_row:
            row_center = (row_start + len(h_projection)) // 2
            row_boundaries.append(row_center)
        
        # Vertical projection: detect columns
        v_projection = np.sum(binary, axis=0)
        v_mean = np.mean(v_projection)
        v_threshold = v_mean * 0.3  # Lower threshold to catch more columns
        
        # Find column boundaries (peaks in vertical projection)
        col_boundaries = []
        in_col = False
        col_start = 0
        
        for i in range(len(v_projection)):
            if v_projection[i] > v_threshold and not in_col:
                col_start = i
                in_col = True
            elif v_projection[i] <= v_threshold and in_col:
                col_end = i
                col_center = (col_start + col_end) // 2
                col_boundaries.append(col_center)
                in_col = False
        
        if in_col:
            col_center = (col_start + len(v_projection)) // 2
            col_boundaries.append(col_center)
        
        # Filter boundaries: keep only significant ones
        # Remove boundaries that are too close together
        min_row_spacing = h * 0.02  # At least 2% of image height
        min_col_spacing = w * 0.02  # At least 2% of image width
        
        filtered_rows = []
        if len(row_boundaries) > 0:
            filtered_rows.append(row_boundaries[0])
            for r in row_boundaries[1:]:
                if r - filtered_rows[-1] > min_row_spacing:
                    filtered_rows.append(r)
        
        filtered_cols = []
        if len(col_boundaries) > 0:
            filtered_cols.append(col_boundaries[0])
            for c in col_boundaries[1:]:
                if c - filtered_cols[-1] > min_col_spacing:
                    filtered_cols.append(c)
        
        # Need at least 2 rows and 2 columns to form a table
        if len(filtered_rows) >= 2 and len(filtered_cols) >= 2:
            # Draw horizontal lines (row separators)
            line_thickness = max(2, int(min(h, w) * 0.001))  # Thickness based on image size
            line_color = (0, 0, 0)  # Black lines
            
            for row_y in filtered_rows:
                cv2.line(result, (0, row_y), (w, row_y), line_color, line_thickness)
            
            # Draw vertical lines (column separators)
            for col_x in filtered_cols:
                cv2.line(result, (col_x, 0), (col_x, h), line_color, line_thickness)
            
            # Draw outer border
            border_thickness = line_thickness * 2
            # Top border
            top_row = min(filtered_rows)
            cv2.line(result, (0, top_row), (w, top_row), line_color, border_thickness)
            # Bottom border
            bottom_row = max(filtered_rows)
            cv2.line(result, (0, bottom_row), (w, bottom_row), line_color, border_thickness)
            # Left border
            left_col = min(filtered_cols)
            cv2.line(result, (left_col, 0), (left_col, h), line_color, border_thickness)
            # Right border
            right_col = max(filtered_cols)
            cv2.line(result, (right_col, 0), (right_col, h), line_color, border_thickness)
            
            logger.info("Bordes de tabla dibujados: %d filas, %d columnas", len(filtered_rows), len(filtered_cols))
        else:
            logger.warning("No se detectó estructura de tabla suficiente (filas: %d, columnas: %d)", len(filtered_rows), len(filtered_cols))
    
    except Exception as e:
        logger.error("Error al dibujar bordes de tabla: %s", e)
    
    return result


def _upscale_image(img: np.ndarray, target_width: Optional[int] = None, target_height: Optional[int] = None, scale_factor: Optional[float] = None) -> np.ndarray:
    """
    Upscale image for better table detection.
    
    Args:
        img: Input image
        target_width: Target width in pixels
        target_height: Target height in pixels
        scale_factor: Scale factor (e.g., 2.0 for 2x)
    
    Returns:
        Upscaled image
    """
    h, w = img.shape[:2]
    
    if scale_factor is not None:
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
    elif target_width is not None and target_height is not None:
        new_w = target_width
        new_h = target_height
    elif target_width is not None:
        scale = target_width / w
        new_w = target_width
        new_h = int(h * scale)
    elif target_height is not None:
        scale = target_height / h
        new_w = int(w * scale)
        new_h = target_height
    else:
        return img
    
    # Use INTER_CUBIC for better quality when upscaling
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def preprocess_for_table_ocr(img: np.ndarray, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """
    Public entrypoint for table OCR preprocessing.
    Uses Tesseract OSD for robust orientation detection, with manual algorithm as fallback.
    Now includes options for Azure Vision table detection enhancement.

    options:
      - enable_deskew: bool (default True) - Apply small-angle deskew correction
      - deskew_max_abs_deg: float (default 8.0) - Maximum deskew angle
      - min_improvement: float (default 0.0) - Minimum improvement for manual algorithm
      - use_tesseract: bool (default True) - Use Tesseract OSD (recommended)
      - enhance_table_structure: bool (default False) - Enhance table borders for Azure Vision
      - enhance_edges: bool (default True) - Enhance table edges (only if enhance_table_structure=True)
      - enhance_contrast: bool (default True) - Enhance contrast (only if enhance_table_structure=True)
      - aggressive_enhancement: bool (default False) - Use aggressive edge enhancement (thicker, more visible borders)
      - detect_rotated_tables: bool (default False) - Detect and rotate table regions that are rotated 90° within the image
      - draw_table_borders: bool (default False) - Detect table structure and draw explicit borders (recommended for Azure Vision)
      - upscale: bool (default False) - Upscale image for better detection
      - upscale_width: int (optional) - Target width for upscaling
      - upscale_height: int (optional) - Target height for upscaling
      - upscale_factor: float (optional) - Scale factor for upscaling (e.g., 2.0)
      - grayscale: bool (default False) - Convert to grayscale (better for table detection)
    """
    options = options or {}

    # Extract rotation-specific options
    enable_deskew = bool(options.get("enable_deskew", True))
    deskew_max_abs_deg = float(options.get("deskew_max_abs_deg", 8.0))
    min_improvement = float(options.get("min_improvement", 0.0))
    use_tesseract = bool(options.get("use_tesseract", True))

    # Apply auto-rotation (Tesseract OSD first, then manual fallback)
    rotated_img, rotation_meta = auto_rotate_table_upright(
        img,
        enable_deskew=enable_deskew,
        deskew_max_abs_deg=deskew_max_abs_deg,
        min_improvement=min_improvement,
        use_tesseract=use_tesseract,
    )
    
    # Build metadata compatible with routes.py expectations
    metadata = {
        "applied_operations": [],
        "rotation_metadata": rotation_meta,
    }
    
    if rotation_meta["auto_rotation_applied"]:
        metadata["applied_operations"].append(f"rotate_{rotation_meta['rotation_degrees']}")
    
    if rotation_meta["deskew_applied"]:
        metadata["applied_operations"].append(f"deskew_{rotation_meta['deskew_angle_deg']:.2f}")
    
    processed = rotated_img
    
    # Log opciones recibidas para debugging
    logger.debug("Opciones de preprocesamiento recibidas: %s", options)
    
    # Detect and rotate table regions that are rotated within the image
    detect_rotated = options.get("detect_rotated_tables", False)
    if detect_rotated:
        processed = _detect_and_rotate_table_region(processed)
        metadata["applied_operations"].append("detect_rotated_tables")
        logger.info("Aplicada detección y rotación de regiones de tabla rotadas")
    
    # Apply grayscale conversion if requested (better for Azure Vision table detection)
    grayscale_enabled = options.get("grayscale", False)
    if grayscale_enabled:
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)  # Keep as BGR for consistency
            metadata["applied_operations"].append("grayscale")
            logger.info("Aplicada conversión a escala de grises")
    
    # Draw explicit table borders (should be done before other enhancements for better detection)
    draw_borders = options.get("draw_table_borders", False)
    if draw_borders:
        processed = _draw_table_borders(processed)
        metadata["applied_operations"].append("draw_table_borders")
        logger.info("Bordes de tabla dibujados explícitamente")
    
    # Apply table structure enhancement for Azure Vision
    enhance_enabled = options.get("enhance_table_structure", False)
    if enhance_enabled:
        enhance_edges = options.get("enhance_edges", True)
        enhance_contrast = options.get("enhance_contrast", True)
        aggressive = options.get("aggressive_enhancement", False)
        processed = _enhance_table_structure(processed, enhance_edges=enhance_edges, enhance_contrast=enhance_contrast, aggressive=aggressive)
        op_name = "enhance_table_structure_aggressive" if aggressive else "enhance_table_structure"
        metadata["applied_operations"].append(op_name)
        logger.info("Aplicada mejora de estructura de tabla (edges=%s, contrast=%s, aggressive=%s)", enhance_edges, enhance_contrast, aggressive)
    
    # Apply upscaling if requested
    upscale_enabled = options.get("upscale", False)
    if upscale_enabled:
        upscale_width = options.get("upscale_width")
        upscale_height = options.get("upscale_height")
        upscale_factor = options.get("upscale_factor")
        
        original_h, original_w = processed.shape[:2]
        processed = _upscale_image(processed, target_width=upscale_width, target_height=upscale_height, scale_factor=upscale_factor)
        new_h, new_w = processed.shape[:2]
        
        scale_info = f"upscale_{new_w}x{new_h}"
        if upscale_factor:
            scale_info = f"upscale_{upscale_factor:.1f}x"
        metadata["applied_operations"].append(scale_info)
        logger.info("Aplicado upscaling: %dx%d -> %dx%d", original_w, original_h, new_w, new_h)
    
    return processed, metadata


def preprocess(img: np.ndarray, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """
    Minimal public entrypoint (alias for compatibility):
    - Only auto-rotation to keep the table upright.

    options:
      - enable_deskew: bool (default True)
      - deskew_max_abs_deg: float (default 8.0)
      - min_improvement: float (default 0.0) - Set to 0 for maximum rotation aggressiveness
    """
    return preprocess_for_table_ocr(img, options)
