import cv2
import numpy as np

def remove_ruled_lines(ink_mask):
    """
    Remove long horizontal ruled lines from ink mask.
    """
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    no_lines = cv2.subtract(ink_mask, detected_lines)
    return no_lines


def enhance_strokes(img, ink_mask):
    """
    Enhance faint ink strokes while preserving original ink color.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Adaptive threshold for fine strokes
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 10
    )

    # Combine with original ink mask
    combined = cv2.bitwise_or(ink_mask, thresh)

    # Close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

    return combined


def extract_signature(image_bytes):
    """
    Extract a real handwritten signature.
    Reject selfies/objects, remove ruled lines, enhance faint strokes.
    Returns: (rgba, confidence, message)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, 0.0, "Invalid image"

    # --- Detect ink regions (black or blue ink) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 30, 30]), np.array([140, 255, 255]))
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 90]))
    ink_mask = cv2.bitwise_or(blue_mask, black_mask)

    # Remove ruled notebook lines
    ink_mask = remove_ruled_lines(ink_mask)

    # Enhance faint strokes
    ink_mask = enhance_strokes(img, ink_mask)

    # --- Early rejection for non-signatures ---
    ink_pixels = cv2.countNonZero(ink_mask)
    total_pixels = img.shape[0] * img.shape[1]
    ink_fraction = ink_pixels / total_pixels

    contours, _ = cv2.findContours(ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stroke_count = sum(1 for c in contours if cv2.contourArea(c) > 10)

    if ink_fraction < 0.0005 or stroke_count < 2:
        return None, 0.0, "Rejected: not enough ink strokes"

    # --- Build final mask ---
    mask = np.zeros_like(ink_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > 10:
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    if cv2.countNonZero(mask) == 0:
        return None, 0.0, "Rejected: no valid signature region"

    # Crop tightly
    ys, xs = np.where(mask > 0)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    roi = img[y_min:y_max + 1, x_min:x_max + 1]
    roi_mask = mask[y_min:y_max + 1, x_min:x_max + 1]

    # Convert to RGBA (transparent background)
    b, g, r = cv2.split(roi)
    rgba = cv2.merge([b, g, r, roi_mask])

    # Confidence score = ink density + stroke count factor
    confidence = min(1.0, (cv2.countNonZero(roi_mask) / roi_mask.size) + (stroke_count / 20))

    return rgba, confidence, "OK"