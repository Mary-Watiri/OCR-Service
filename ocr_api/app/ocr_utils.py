import tempfile
import cv2
import easyocr
import numpy as np
import re
import json
from typing import List, Dict
import logging
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Initialize EasyOCR
reader = easyocr.Reader(['en', 'sw'], gpu=False)

# Function to convert image bytes to image object
def byte_to_image(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    return image

def adaptive_resize(image, resize_factor=2.0, max_size=None, min_size=None):
    h, w = image.shape[:2]
    
    if max_size:
        scale_factor = max_size / float(max(h, w))
        if scale_factor < 1:
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    if min_size:
        scale_factor = min_size / float(min(h, w))
        if scale_factor > 1:
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Default resizing logic based on image dimensions
    if max(h, w) < 1000:
        resize_factor = 3.0
    else:
        resize_factor = 1.5
    
    # Final resize with the calculated resize_factor
    return cv2.resize(image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)

# def preprocess_image(image):
#     # Step 1: Adaptive resizing to improve text clarity
#     image = adaptive_resize(image)

#     # Step 2: Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Step 3: Reduce noise while preserving edges
#     blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

#     # Step 4: Invert and threshold the image for contour detection
#     gray_inv = cv2.bitwise_not(blurred)
#     _, thresh = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#     # Step 5: Skew correction
#     coords = np.column_stack(np.where(thresh > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle

#     # Rotate the image to deskew
#     (h, w) = image.shape[:2]
#     M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
#     rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#     # Step 6: Apply CLAHE to enhance contrast
#     gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     contrast_enhanced = clahe.apply(gray_rotated)

#     # Step 7: Convert back to 3 channels if needed by downstream models (e.g., CRAFT)
#     final_img = cv2.cvtColor(contrast_enhanced, cv2.COLOR_GRAY2BGR)

#     return final_img

def preprocess_image(image):
    # Step 1: Adaptive resizing to balance detail and processing
    image = adaptive_resize(image, max_size=1200, min_size=600)

    # Step 2: Perspective correction using edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 1000:
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]  
                rect[2] = pts[np.argmax(s)]  
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]  
                rect[3] = pts[np.argmax(diff)]  
                
                (tl, tr, br, bl) = rect
                width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                max_width = max(int(width_a), int(width_b))
                height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                max_height = max(int(height_a), int(height_b))
                
                dst = np.array([
                    [0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1]], dtype="float32")
                
                M = cv2.getPerspectiveTransform(rect, dst)
                image = cv2.warpPerspective(image, M, (max_width, max_height))

    # Step 3: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 4: Light Gaussian blur to reduce noise while preserving text
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 5: CLAHE for localized contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Step 6: Return high-contrast grayscale image for EasyOCR
    return enhanced

# Example usage for testing
def extract_text_from_image(image):
    results = reader.readtext(image, detail=1)
    text = [res[1] for res in results if res[2] > 0.5]
    logging.info(f"Extracted text: {text}")
    return text

def extract_fields(ocr_text):
    field_labels = {
        "SERIAL NUMBER": "Serial Number",
        "ID NUMBER": "ID Number",
        "ID NO": "ID Number",
        "FULL NAMES": "Full Names",
        "NAMES": "Full Names",
        "SEX": "Sex",
        "GENDER": "Sex",
        "DISTRICT OF BIRTH": "Place of Birth",
        "PLACE OF ISSUE": "Place of Issue",
        "DATE OF ISSUE": "Date of Issue",
        "HOLDER'S SIGN": "Holder's Signature",
        "SIGNATURE": "Holder's Signature",
        "NATIONALITY": "Nationality",
    }

    def is_date(text):
        return re.match(r'\d{2}[./-]\d{2}[./-]\d{4}', text.replace(" ", "")) is not None

    cleaned_text = [w.strip(" :,.").upper() for w in ocr_text if w.strip()]
    logging.info(f"Cleaned OCR Text: {cleaned_text}")

    fields = {}
    idx = 0
    while idx < len(cleaned_text):
        word = cleaned_text[idx]

        # Date of Birth
        if "FULL NAMES" in word and idx + 3 < len(cleaned_text):
            surname = cleaned_text[idx + 1]
            maybe_given = cleaned_text[idx + 2]
            maybe_dob = cleaned_text[idx + 3]

            fields["Surname"] = surname
            if is_date(maybe_dob):
                fields["Given Names"] = maybe_given
                fields["Date of Birth"] = maybe_dob.replace(" ", "").replace(".", "-").replace("/", "-")
                idx += 3
            else:
                fields["Given Names"] = f"{maybe_given} {maybe_dob}"
                if idx + 4 < len(cleaned_text) and is_date(cleaned_text[idx + 4]):
                    fields["Date of Birth"] = cleaned_text[idx + 4].replace(" ", "").replace(".", "-").replace("/", "-")
                    idx += 4
                else:
                    idx += 2
        elif word in field_labels:
            label = field_labels[word]
            if idx + 1 < len(cleaned_text):
                next_word = cleaned_text[idx + 1]
                if is_date(next_word):
                    fields[label] = next_word.replace(" ", "").replace(".", "-").replace("/", "-")
                else:
                    fields[label] = next_word
                idx += 1
        idx += 1

    if "Surname" in fields and "Given Names" in fields:
        fields["Full Name"] = f"{fields['Surname']} {fields['Given Names']}".strip()

    return fields

def normalize_date(text):
    return text.replace(" ", "").replace(".", "-").replace("/", "-")

def is_date(text: str) -> bool:
    return bool(re.match(r'^\d{2}[./-]\d{2}[./-]\d{4}$', text))

def is_word(word: str) -> bool:
    return bool(re.match(r"^[A-Za-z]+$", word))

def is_id_number(word: str) -> bool:
    return bool(re.match(r"^\d{12}$", word))  


def extract_maisha_card_fields(ocr_text):
    fields = {}
    cleaned_text = [word.upper().replace(' ', '') for word in ocr_text if word.strip()]
    logging.info(f"Cleaned OCR Text for Maisha Card: {cleaned_text}")

    # Extract dates
    dates = [word.replace(' ', '') for word in cleaned_text if re.match(r'\d{2}\.\d{2}\.\d{4}', word.replace(' ', ''))]
    if len(dates) >= 1:
        fields["Date of Birth"] = dates[0]
    if len(dates) >= 2:
        fields["Expiry Date"] = dates[1]

    # Extract ID number: usually a 9-digit number
    id_number = next((word for word in cleaned_text if re.match(r'^\d{8,10}$', word)), None)
    if id_number:
        fields["ID Number"] = id_number

    # Extract gender
    gender = next((word for word in cleaned_text if word in ["MALE", "FEMALE"]), None)
    if gender:
        fields["Gender"] = gender

    # Extract nationality
    if "KEN" in cleaned_text or "KENYAN" in cleaned_text:
        fields["Nationality"] = "Kenyan"

    # Extract names (before gender)
    if gender:
        gender_index = cleaned_text.index(gender)
        possible_names = cleaned_text[:gender_index]
        fields["Full Names"] = " ".join(possible_names).title()

    # Determine place of birth / issue using context
    place_candidates = [word for word in cleaned_text if word.isalpha() and word.isupper() and word not in fields.values()]
    for idx, word in enumerate(cleaned_text):
        if word in place_candidates:
            if "Date of Birth" in fields and word in cleaned_text:
                dob_index = cleaned_text.index(fields["Date of Birth"])
                expiry_index = cleaned_text.index(fields["Expiry Date"]) if "Expiry Date" in fields else len(cleaned_text)
                word_index = cleaned_text.index(word)

                if dob_index < word_index < expiry_index:
                    fields["Place of Birth"] = word.title()
                elif word_index > expiry_index:
                    fields["Place of Issue"] = word.title()

    logging.info(f"Final extracted Maisha Card fields: {fields}")
    return fields

def process_id_image(image_bytes):
    try:
        # Convert bytes to image
        image = byte_to_image(image_bytes)
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Extract text using EasyOCR
        ocr_text = extract_text_from_image(processed_image)
        
        # Extract structured fields
        fields = extract_fields(ocr_text)
        
        return json.dumps(fields, indent=2)
    except Exception as e:
        logging.error(f"Error processing ID image: {str(e)}")
        return json.dumps({"error": str(e)})

def process_maisha_card_image(image_bytes):
    try:
        # Convert bytes to image
        image = byte_to_image(image_bytes)
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Extract text using EasyOCR
        ocr_text = extract_text_from_image(processed_image)
        
        logging.info(f"Extracted text: {ocr_text}")
        
        # Extract structured fields
        fields = extract_maisha_card_fields(ocr_text)
        
        logging.info(f"Final extracted Maisha Card fields: {fields}")
        
        return json.dumps(fields, indent=2)
    except Exception as e:
        logging.error(f"Error processing Maisha card image: {str(e)}")
        return json.dumps({"error": str(e)})


