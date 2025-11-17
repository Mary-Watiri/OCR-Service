import cv2
import numpy as np

# -------------------------------------------------------------------
# Remove ruled or document-like lines
# -------------------------------------------------------------------
def remove_ruled_lines(ink_mask):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    h_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, horizontal_kernel)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
    v_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, vertical_kernel)

    detected_lines = cv2.bitwise_or(h_lines, v_lines)
    return cv2.subtract(ink_mask, detected_lines)


# -------------------------------------------------------------------
# Enhance faint strokes
# -------------------------------------------------------------------
def enhance_strokes(img, ink_mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    t1 = cv2.adaptiveThreshold(enhanced, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 5)
    _, t2 = cv2.threshold(enhanced, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    final = cv2.bitwise_or(t1, t2)
    final = cv2.bitwise_or(final, ink_mask)

    kernel = np.ones((2, 2), np.uint8)
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel)
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel)

    return final


# -------------------------------------------------------------------
# Reject obviously non-signature images
# -------------------------------------------------------------------
def detect_and_reject_non_signatures(img, ink_mask):
    height, width = img.shape[:2]
    ink_pixels = cv2.countNonZero(ink_mask)
    frac = ink_pixels / (height * width)

    if frac > 0.80:
        return False, "Rejected: almost full-ink image"
    if frac < 0.00005:
        return False, "Rejected: too little ink"

    contours, _ = cv2.findContours(ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 8]

    if len(areas) > 12:
        std = np.std(areas)
        mean = np.mean(areas)
        if mean > 0 and std / mean < 0.35:
            return False, "Rejected: appears to be printed/written text"

    return True, "OK"


# -------------------------------------------------------------------
# Signature-likeness score
# -------------------------------------------------------------------
def detect_signature_like_features(img, ink_mask):
    ys, xs = np.where(ink_mask > 0)
    if len(xs) == 0:
        return 0

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    w, h = x2 - x1, y2 - y1
    if w == 0 or h == 0:
        return 0

    aspect = max(w / h, h / w)
    if aspect > 10:
        return 0.25

    region_area = w * h
    ink = cv2.countNonZero(ink_mask)
    density = ink / region_area

    if density < 0.01:
        density_score = 0.15
    elif density > 0.90:
        density_score = 0.25
    else:
        density_score = min(1.0, density * 1.2)

    dist = cv2.distanceTransform(ink_mask, cv2.DIST_L2, 5)
    sw = dist[dist > 0]
    if len(sw) == 0:
        return 0.2

    sw_std = np.std(sw)
    stroke_score = min(1.0, (sw_std + 1) / 3)

    return max(0.0, min(1.0,
        0.45 * density_score +
        0.55 * stroke_score
    ))


# -------------------------------------------------------------------
# Remove phone shadows safely
# -------------------------------------------------------------------
def remove_phone_shadow(img, ink_mask=None):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Smooth the image to get large-scale illumination
    smooth_L = cv2.GaussianBlur(L, (251, 251), 0)
    shadow_map = smooth_L.astype(np.int32) - L.astype(np.int32)
    shadow_map = cv2.normalize(shadow_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, shadow_mask = cv2.threshold(shadow_map, 40, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)

    h, w = shadow_mask.shape
    min_area = (h * w) * 0.03
    contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(shadow_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    # Prepare correction
    corrected_L = L.copy()
    correction = cv2.GaussianBlur(shadow_map, (251, 251), 0)
    correction = cv2.normalize(correction, None, 0, 80, cv2.NORM_MINMAX).astype(np.uint8)

    # Avoid brightening ink
    if ink_mask is not None:
        clean_mask = cv2.bitwise_and(clean_mask, cv2.bitwise_not(ink_mask))

    # Apply correction safely
    corrected_L = np.where(clean_mask > 0,
                           cv2.add(corrected_L, correction),
                           corrected_L)

    corrected_L = np.clip(corrected_L, 0, 255).astype(np.uint8)
    corrected_img = cv2.cvtColor(cv2.merge([corrected_L, A, B]), cv2.COLOR_LAB2BGR)

    return corrected_img, clean_mask


# -------------------------------------------------------------------
# Main extractor
# -------------------------------------------------------------------
def extract_signature(image_bytes):
    data = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None, 0, "Invalid image", None

    # Step 0: Broad ink detection for shadow protection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (80, 10, 10), (150, 255, 255))
    black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 150))
    red1 = cv2.inRange(hsv, (0, 20, 20), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 20, 20), (180, 255, 255))
    ink_mask = blue | black | red1 | red2

    # Step 1: Remove shadows
    img, shadow_mask = remove_phone_shadow(img, ink_mask=ink_mask)

    # Step 2: Recompute ink mask after correction
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (80, 10, 10), (150, 255, 255))
    black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 150))
    red1 = cv2.inRange(hsv, (0, 20, 20), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 20, 20), (180, 255, 255))
    ink_mask = blue | black | red1 | red2

    # Step 3: Remove ruled lines & enhance strokes
    ink_mask = remove_ruled_lines(ink_mask)
    ink_mask = enhance_strokes(img, ink_mask)

    # Step 4: Reject non-signature images
    ok, reason = detect_and_reject_non_signatures(img, ink_mask)
    if not ok:
        return None, 0, reason, shadow_mask

    # Step 5: Signature-likeness score
    score = detect_signature_like_features(img, ink_mask)
    if score < 0.25:
        return None, score, "Rejected: weak signature characteristics", shadow_mask

    ys, xs = np.where(ink_mask > 0)
    if len(xs) == 0:
        return None, 0, "No ink region", shadow_mask

    y1, y2 = max(0, ys.min() - 5), min(img.shape[0], ys.max() + 5)
    x1, x2 = max(0, xs.min() - 5), min(img.shape[1], xs.max() + 5)

    roi = img[y1:y2, x1:x2]
    mask_roi = ink_mask[y1:y2, x1:x2]
    _, mask_roi = cv2.threshold(mask_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    b, g, r = cv2.split(roi)
    rgba = cv2.merge([b, g, r, mask_roi])

    confidence = (0.4 * score +
                  0.4 * min(1, cv2.countNonZero(mask_roi) / (mask_roi.size * 0.4)) +
                  0.2)

    return rgba, float(min(confidence, 1)), "OK", shadow_mask
