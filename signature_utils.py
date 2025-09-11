import cv2
import numpy as np
from sklearn.cluster import KMeans

def remove_ruled_lines(ink_mask):
    """
    Remove long horizontal and vertical ruled lines from ink mask.
    """
    # Horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_h_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # Vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_v_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    # Combine detected lines
    detected_lines = cv2.bitwise_or(detected_h_lines, detected_v_lines)
    
    # Remove lines from mask
    no_lines = cv2.subtract(ink_mask, detected_lines)
    return no_lines

def detect_signature_like_features(img, ink_mask):
    """
    Enhanced signature detection with better feature analysis.
    Returns a score indicating how signature-like the image is.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate aspect ratio of ink region
    ys, xs = np.where(ink_mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        return 0.0
    
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    width = x_max - x_min
    height = y_max - y_min
    
    if width == 0 or height == 0:
        return 0.0
    
    aspect_ratio = max(width/height, height/width)
    
    # Signature typically has aspect ratio between 1:4 and 4:1
    # But be more lenient for various signature styles
    if aspect_ratio > 6:
        return 0.3  # Very elongated, might still be signature
    
    # Calculate density (how concentrated the ink is)
    ink_region_area = width * height
    if ink_region_area == 0:
        return 0.0
        
    ink_pixels = cv2.countNonZero(ink_mask)
    density = ink_pixels / ink_region_area
    
    # Signatures typically have moderate density (not too sparse, not too dense)
    # Broaden the acceptable range
    if density < 0.05:  # Very sparse
        return 0.4
    if density > 0.85:  # Very dense
        return 0.4
    
    # Calculate stroke width variation (signatures have varying stroke widths)
    dist_transform = cv2.distanceTransform(ink_mask, cv2.DIST_L2, 5)
    stroke_width_values = dist_transform[dist_transform > 0]
    
    if len(stroke_width_values) == 0:
        return 0.0
        
    stroke_width_std = np.std(stroke_width_values)
    stroke_width_mean = np.mean(stroke_width_values)
    
    # More lenient stroke width analysis
    if stroke_width_std < 0.3 and stroke_width_mean < 2.0:
        return 0.5  # Could still be a signature with uniform strokes
    
    # Calculate curvature and complexity
    contours, _ = cv2.findContours(ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
        
    # Calculate various signature characteristics
    complexity_score = 0
    curvature_score = 0
    stroke_count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5:  # Smaller threshold for faint signatures
            stroke_count += 1
            
            # Compactness (area/perimeter^2)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter * perimeter)
                # Signatures are less compact than text characters
                if compactness < 0.3:
                    complexity_score += 0.8
                elif compactness < 0.5:
                    complexity_score += 0.5
                else:
                    complexity_score += 0.2
            
            # Curvature analysis
            if len(cnt) > 4:
                # Fit ellipse to assess curvature
                ellipse = cv2.fitEllipse(cnt)
                (center, axes, orientation) = ellipse
                major_axis, minor_axis = max(axes), min(axes)
                if minor_axis > 0:
                    eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2)
                    curvature_score += eccentricity * 0.5
    
    # Normalize scores
    if stroke_count > 0:
        complexity_score /= stroke_count
        curvature_score /= stroke_count
    else:
        complexity_score = 0
        curvature_score = 0
    
    # Combine features into a final score with more balanced weights
    signature_score = min(1.0, 
        0.2 * min(1.0, density/0.6) +  # Density contribution
        0.2 * min(1.0, stroke_width_std/3.0) +  # Stroke variation
        0.3 * complexity_score +  # Shape complexity
        0.2 * curvature_score +  # Curvature
        0.1 * min(1.0, stroke_count/15.0)  # Number of strokes
    )
    
    return signature_score

def detect_and_reject_non_signatures(img, ink_mask):
    """
    More sophisticated non-signature detection with better thresholds.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    
    # 1. Check ink coverage percentage
    ink_pixels = cv2.countNonZero(ink_mask)
    total_pixels = height * width
    ink_fraction = ink_pixels / total_pixels
    
    # More lenient thresholds for signatures
    if ink_fraction > 0.4:  # Too much ink (could be photo or document)
        return False, "Rejected: excessive ink coverage"
    if ink_fraction < 0.0001:  # Almost no ink
        return False, "Rejected: insufficient ink"
    
    # 2. Check if this might be structured text
    contours, _ = cv2.findContours(ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours and len(contours) > 3:
        # Calculate size and spacing regularity
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 5]
        if areas and len(areas) > 5:
            area_std = np.std(areas)
            area_mean = np.mean(areas)
            
            # Text has more uniform character sizes than signatures
            if area_std / area_mean < 0.4 and len(areas) > 8:
                return False, "Rejected: appears to be structured text"
    
    # 3. Check for faces (but be careful with small images)
    if height > 100 and width > 100:  # Only check reasonable sized images
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            if len(faces) > 0:
                return False, "Rejected: contains faces"
        except:
            pass  # Skip face detection if it fails
    
    # 4. Check for document structure (lines, grids)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    
    if lines is not None and len(lines) > 5:
        horizontal_lines = 0
        vertical_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if abs(angle) < 15 or abs(angle - 180) < 15:
                horizontal_lines += 1
            elif abs(angle - 90) < 15 or abs(angle - 270) < 15:
                vertical_lines += 1
        
        # If many structured lines, likely a document
        if horizontal_lines > 4 and vertical_lines > 4:
            return False, "Rejected: appears to be a structured document/form"
    
    # 5. Check color variation (signatures usually have consistent color)
    if len(img.shape) == 3:
        ink_pixels_rgb = img[ink_mask > 0]
        if len(ink_pixels_rgb) > 10:
            # Check if multiple distinct colors are present
            color_std = np.std(ink_pixels_rgb, axis=0)
            avg_color_std = np.mean(color_std)
            if avg_color_std > 50:  # High color variation
                return False, "Rejected: too many colors for a signature"
    
    return True, "OK"

def enhance_strokes(img, ink_mask):
    """
    Enhanced stroke enhancement for faint signatures.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Multiple thresholding techniques
    # Adaptive threshold
    thresh_adapt = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 7
    )
    
    # Otsu's threshold
    _, thresh_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combine thresholds
    combined_thresh = cv2.bitwise_or(thresh_adapt, thresh_otsu)
    
    # Combine with original ink mask
    enhanced_mask = cv2.bitwise_or(ink_mask, combined_thresh)
    
    # Remove noise
    kernel = np.ones((2, 2), np.uint8)
    enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Fill small holes
    enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return enhanced_mask

def extract_signature(image_bytes):
    """
    Enhanced signature extraction with better signature recognition.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, 0.0, "Invalid image"

    # --- Detect ink regions with multiple color ranges ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Broader color ranges for various ink colors
    # Blue ink
    blue_mask = cv2.inRange(hsv, np.array([85, 20, 20]), np.array([145, 255, 255]))
    
    # Black ink (broader range)
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 120]))
    
    # Red ink
    red_mask1 = cv2.inRange(hsv, np.array([0, 20, 20]), np.array([15, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([155, 20, 20]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Green ink
    green_mask = cv2.inRange(hsv, np.array([30, 20, 20]), np.array([95, 255, 255]))
    
    # Combine all ink masks
    ink_mask = cv2.bitwise_or(blue_mask, black_mask)
    ink_mask = cv2.bitwise_or(ink_mask, red_mask)
    ink_mask = cv2.bitwise_or(ink_mask, green_mask)

    # Remove ruled notebook lines
    ink_mask = remove_ruled_lines(ink_mask)

    # Enhance faint strokes
    ink_mask = enhance_strokes(img, ink_mask)

    # --- Check if this is likely a signature ---
    is_signature, rejection_reason = detect_and_reject_non_signatures(img, ink_mask)
    if not is_signature:
        return None, 0.0, rejection_reason

    # Calculate signature-like features score
    signature_score = detect_signature_like_features(img, ink_mask)
    
    # More lenient threshold for signature recognition
    if signature_score < 0.35:
        return None, signature_score, "Rejected: doesn't resemble a signature"

    # --- Build final mask ---
    contours, _ = cv2.findContours(ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stroke_count = sum(1 for c in contours if cv2.contourArea(c) > 5)
    
    if stroke_count < 2:
        return None, 0.0, "Rejected: not enough ink strokes"

    mask = np.zeros_like(ink_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > 5:  # Smaller area threshold for faint signatures
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    if cv2.countNonZero(mask) == 0:
        return None, 0.0, "Rejected: no valid signature region"

    # Crop tightly to signature
    ys, xs = np.where(mask > 0)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    
    # Add some padding
    padding = 5
    y_min = max(0, y_min - padding)
    y_max = min(img.shape[0] - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(img.shape[1] - 1, x_max + padding)
    
    roi = img[y_min:y_max + 1, x_min:x_max + 1]
    roi_mask = mask[y_min:y_max + 1, x_min:x_max + 1]

    # Convert to RGBA (transparent background)
    b, g, r = cv2.split(roi)
    rgba = cv2.merge([b, g, r, roi_mask])

    # Enhanced confidence calculation
    density_score = min(1.0, cv2.countNonZero(roi_mask) / max(1, roi_mask.size))
    stroke_score = min(1.0, stroke_count / 12.0)
    size_score = min(1.0, (roi_mask.size) / (100 * 100))  # Normalize by reasonable size
    
    confidence = min(1.0, 
        0.25 * density_score + 
        0.25 * stroke_score + 
        0.4 * signature_score + 
        0.1 * size_score
    )

    return rgba, confidence, "OK"