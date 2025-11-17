import cv2
import numpy as np
from skimage.morphology import skeletonize

# ------------------------------
# Remove ruled lines
# ------------------------------
def remove_ruled_lines(ink_mask):
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
    h_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, v_kernel)
    lines = cv2.bitwise_or(h_lines, v_lines)
    return cv2.subtract(ink_mask, lines)

# ------------------------------
# Enhance faint strokes
# ------------------------------
def enhance_strokes(img, ink_mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    t1 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 5)
    _, t2 = cv2.threshold(enhanced, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    combined = cv2.bitwise_or(t1, t2)
    combined = cv2.bitwise_or(combined, ink_mask)

    kernel = np.ones((2,2), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    return combined

# ------------------------------
# Shadow removal
# ------------------------------
def remove_phone_shadow(img, ink_mask=None):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    smooth_L = cv2.GaussianBlur(L, (251, 251), 0)
    shadow_map = smooth_L.astype(np.int32) - L.astype(np.int32)
    shadow_map = cv2.normalize(shadow_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, shadow_mask = cv2.threshold(shadow_map, 40, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45,45))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)

    h, w = shadow_mask.shape
    min_area = (h*w)*0.03
    contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(shadow_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
    if ink_mask is not None:
        clean_mask = cv2.bitwise_and(clean_mask, cv2.bitwise_not(ink_mask))

    correction = cv2.GaussianBlur(shadow_map, (251,251),0)
    correction = cv2.normalize(correction, None, 0, 80, cv2.NORM_MINMAX).astype(np.uint8)

    mask_bool = (clean_mask>0) & (L<220)
    L = np.where(mask_bool, cv2.add(L, correction), L)
    L = np.clip(L, 0, 255).astype(np.uint8)
    corrected_img = cv2.cvtColor(cv2.merge([L,A,B]), cv2.COLOR_LAB2BGR)
    return corrected_img, clean_mask

# ------------------------------
# Reject non-signatures robustly
# ------------------------------
def reject_non_signatures(img, ink_mask):
    h, w = img.shape[:2]
    ink_frac = cv2.countNonZero(ink_mask) / (h*w)

    if ink_frac > 0.8:
        return False, "Too much ink: likely photo"
    if ink_frac < 0.0005:
        return False, "Too little ink"

    contours, _ = cv2.findContours(ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area > 5000:  # large blobs indicate photo/texture
            return False, "Large blob detected"
        x,y,wc,hc = cv2.boundingRect(c)
        aspect = wc/hc if hc>0 else 0
        if aspect>15 or aspect<0.05:
            return False, "Extreme contour aspect ratio"

    # Skeleton analysis
    skel = skeletonize((ink_mask>0).astype(np.uint8))
    skel_pixels = np.count_nonzero(skel)
    if skel_pixels < 50 or skel_pixels > ink_mask.size*0.4:
        return False, "Skeleton indicates non-signature"

    return True, "OK"

# ------------------------------
# Signature-likeness score
# ------------------------------
def signature_score(ink_mask):
    ys, xs = np.where(ink_mask>0)
    if len(xs)==0:
        return 0
    x1,x2,y1,y2 = xs.min(), xs.max(), ys.min(), ys.max()
    w,h = x2-x1, y2-y1
    if w==0 or h==0:
        return 0

    aspect = w/h
    if aspect<0.5 or aspect>10:
        return 0.2

    density = cv2.countNonZero(ink_mask)/(w*h)
    density_score = min(1.0, max(0.1, density*1.2))

    dist = cv2.distanceTransform(ink_mask, cv2.DIST_L2,5)
    sw = dist[dist>0]
    stroke_score = min(1.0, (np.std(sw)+1)/3) if len(sw)>0 else 0.2

    return 0.45*density_score + 0.55*stroke_score

# ------------------------------
# Main extractor
# ------------------------------
def extract_signature(image_bytes):
    data = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None, 0, "Invalid image", None

    # Step 0: initial ink mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (90,50,50), (130,255,255))
    black = cv2.inRange(hsv, (0,0,0), (180,255,120))
    ink_mask = cv2.bitwise_or(blue, black)

    # Step 1: remove shadows
    img, shadow_mask = remove_phone_shadow(img, ink_mask=ink_mask)

    # Step 2: recompute ink mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (90,50,50), (130,255,255))
    black = cv2.inRange(hsv, (0,0,0), (180,255,120))
    ink_mask = cv2.bitwise_or(blue, black)

    # Step 3: clean & enhance ink
    ink_mask = remove_ruled_lines(ink_mask)
    ink_mask = enhance_strokes(img, ink_mask)

    # Step 4: reject non-signatures
    ok, reason = reject_non_signatures(img, ink_mask)
    if not ok:
        return None, 0, reason, shadow_mask

    # Step 5: signature-likeness score
    score = signature_score(ink_mask)
    if score < 0.35:
        return None, score, "Rejected: not signature-like", shadow_mask

    # Step 6: extract signature region
    ys,xs = np.where(ink_mask>0)
    y1,y2 = max(0, ys.min()-5), min(img.shape[0], ys.max()+5)
    x1,x2 = max(0, xs.min()-5), min(img.shape[1], xs.max()+5)

    roi = img[y1:y2, x1:x2]
    mask_roi = ink_mask[y1:y2, x1:x2]
    _, mask_roi = cv2.threshold(mask_roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    b,g,r = cv2.split(roi)
    rgba = cv2.merge([b,g,r,mask_roi])

    confidence = min(1.0, 0.3*score + 0.5*min(1, cv2.countNonZero(mask_roi)/(mask_roi.size*0.4)) + 0.2)

    return rgba, float(confidence), "OK", shadow_mask
