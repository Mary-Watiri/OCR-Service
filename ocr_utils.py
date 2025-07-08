import tempfile
import cv2
import easyocr
import calendar
import numpy as np
import re
import json
from datetime import datetime
from typing import List, Dict
from passporteye import read_mrz
import logging
import unicodedata
from typing import Optional


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
    fields = {}
    field_sources = {}

    def clean_text(text):
        return text.strip(" :,.").upper()

    cleaned_text = [clean_text(word) for word in ocr_text if word.strip()]
    logging.info(f"Cleaned OCR Text for Kenyan ID: {cleaned_text}")

    def is_date(text):
        return bool(re.match(r'\d{2}[./-]\d{2}[./-]\d{4}', text.replace(" ", "")))

    def is_id_number(text):
        return bool(re.match(r'^\d{8,10}$', text))

    def is_name_part(text):
        return bool(re.match(r'^[A-Z]+$', text)) and text not in {"SEX", "MALE", "FEMALE", "X", "HALE"}

    def infer_sex(text):
        text_upper = text.upper()
        if text_upper in {"MALE", "X", "HALE", "ALE", "MA"} or (text_upper.startswith("M") and len(text_upper) == 4):
            return "Male"
        if text_upper in {"FEMALE", "FMALE", "FEMLE", "FEM"} or (text_upper.startswith("F") and len(text_upper) >= 5):
            return "Female"
        return None

    def fuzzy_label_match(text, labels):
        text_clean = text.replace(" ", "")
        for label in labels:
            if label.replace(" ", "") in text_clean or text_clean in label.replace(" ", ""):
                return labels[label]
        return None

    field_labels = {
        "SERIAL NUMBER": "Serial Number",
        "SERIALNUNBER": "SerialNumber",
        "ID NUMBER": "ID Number",
        "FULL NAMES": "Full Names",
        "DATE OF BIRTH": "Date of Birth",
        "SEX": "Sex",
        "DISTRICT OF BIRTH": "District of Birth",
        "PLACE OF ISSUE": "Place of Issue",
        "DATE OF ISSUE": "Date of Issue",
        "HOLDER'S SIGN": "Holder's Signature",
    }

    idx = 0
    dates_collected = []
    numbers_collected = []

    while idx < len(cleaned_text):
        word = cleaned_text[idx]
        matched_label = fuzzy_label_match(word, field_labels)

        if matched_label:
            label = matched_label
            if idx + 1 < len(cleaned_text):
                next_word = cleaned_text[idx + 1]

                if label in {"Date of Birth", "Date ofIssue"}:
                    if is_date(next_word):
                        date_val = next_word.replace(" ", "").replace(".", "-").replace("/", "-")
                        fields[label] = date_val
                        field_sources[label] = f"Matched label: {word} + valid date"
                        dates_collected.append(next_word)
                    else:
                        fields[label] = next_word
                        field_sources[label] = f"Matched label: {word} + fallback"
                    idx += 1

                elif label == "Full Names":
                    name_parts = []
                    temp_idx = idx + 1
                    while temp_idx < len(cleaned_text):
                        next_part = cleaned_text[temp_idx]
                        if is_date(next_part) or fuzzy_label_match(next_part, field_labels):
                            break
                        if is_name_part(next_part):
                            name_parts.append(next_part)
                        temp_idx += 1
                    if name_parts:
                        fields[label] = " ".join(name_parts).title()
                        field_sources[label] = f"Matched label: {word} + name parts"
                    idx = temp_idx - 1

                elif label == "Sex":
                    inferred = infer_sex(next_word)
                    fields[label] = inferred if inferred else next_word
                    field_sources[label] = f"Matched label: {word} + inferred sex"
                    idx += 1

                elif label in {"Serial Number", "ID Number"}:
                    if is_id_number(next_word):
                        fields[label] = next_word
                        field_sources[label] = f"Matched label: {word} + valid ID"
                        numbers_collected.append(next_word)
                    else:
                        fields[label] = next_word
                        field_sources[label] = f"Matched label: {word} + fallback"
                    idx += 1

                else:
                    fields[label] = next_word
                    field_sources[label] = f"Matched label: {word}"
                    idx += 1

        elif is_date(word):
            dates_collected.append(word)
        elif is_id_number(word):
            numbers_collected.append(word)
        idx += 1

    # Fallback: Dates
    if dates_collected:
        if "Date of Birth" not in fields:
            fields["Date of Birth"] = dates_collected[0].replace(" ", "").replace(".", "-").replace("/", "-")
            field_sources["Date of Birth"] = "Fallback from collected dates"
        if len(dates_collected) > 1 and "Date of Issue" not in fields:
            fields["Date of Issue"] = dates_collected[1].replace(" ", "").replace(".", "-").replace("/", "-")
            field_sources["Date of Issue"] = "Fallback from collected dates"

    # Fallback: Numbers
    if numbers_collected:
        if "Serial Number" not in fields:
            fields["Serial Number"] = numbers_collected[0]
            field_sources["Serial Number"] = "Fallback from collected numbers"
        if len(numbers_collected) > 1 and "ID Number" not in fields:
            fields["ID Number"] = numbers_collected[1]
            field_sources["ID Number"] = "Fallback from collected numbers"

    # Full Names extraction with fallback
    try:
        start = cleaned_text.index("FULL NAMES") + 1
        stop_labels = {"DATE OF BIRTH", "SEX", "DISTRICT OF BIRTH"}
        end = next((i for i in range(start, len(cleaned_text)) if cleaned_text[i] in stop_labels), len(cleaned_text))
        name_parts = cleaned_text[start:end]
        if name_parts:
            full_name = " ".join(name_parts).title()
            fields["Full Names"] = full_name
            field_sources["Full Names"] = "Matched label: FULL NAMES + name parts"
    except ValueError:
        # Fallback: Scan through cleaned text to find best candidate
        possible_names = []
        skip_words = set(field_labels.keys()) | {"KENYA", "JAMHURI", "REPUBLIC", "OF"}

        for word in cleaned_text:
            word_parts = word.split()
            if any(part in skip_words for part in word_parts):
                continue
            if all(is_name_part(part) for part in word_parts) and len(word_parts) >= 2:
                possible_names.append(word_parts)

        if possible_names:
            best_name = max(possible_names, key=len)
            fields["Full Names"] = " ".join(best_name).title()
            field_sources["Full Names"] = "Fallback from filtered longest match"

    # Fallback: Sex
    if "Sex" not in fields:
        for word in cleaned_text:
            inferred = infer_sex(word)
            if inferred:
                fields["Sex"] = inferred
                field_sources["Sex"] = f"Fallback from inferred sex: {word}"
                break

    # Heuristic: District of Birth & Place of Issue
    dob_date = fields.get("Date of Birth", "").replace("-", ".")
    doi_date = fields.get("Date of Issue", "").replace("-", ".")
    for idx, word in enumerate(cleaned_text):
        if is_name_part(word) and word not in fields.values() and word not in fields.get("Full Names", "").upper().split():
            if "District of Birth" not in fields and dob_date and dob_date in cleaned_text:
                dob_index = cleaned_text.index(dob_date)
                if idx > dob_index:
                    fields["District of Birth"] = word.title()
                    field_sources["District of Birth"] = f"Heuristic after Date of Birth: {word}"
                    continue
            if "Place of Issue" not in fields and doi_date and doi_date in cleaned_text:
                doi_index = cleaned_text.index(doi_date)
                if idx > doi_index:
                    fields["Place of Issue"] = word.title()
                    field_sources["Place of Issue"] = f"Heuristic after Date of Issue: {word}"

    # Fallback: Based on position after SEX
    if "District of Birth" not in fields or "Place of Issue" not in fields:
        try:
            sex_index = None
            if "FEMALE" in cleaned_text:
                sex_index = cleaned_text.index("FEMALE")
            elif "MALE" in cleaned_text:
                sex_index = cleaned_text.index("MALE")

            if sex_index is not None:
                location_candidates = []
                for i in range(sex_index + 1, min(len(cleaned_text), sex_index + 5)):
                    val = cleaned_text[i]
                    if all(part.isalpha() for part in val.split()):
                        location_candidates.append(val.title())

                if "District of Birth" not in fields and len(location_candidates) >= 1:
                    fields["District of Birth"] = location_candidates[0]
                    field_sources["District of Birth"] = "Based on position after SEX"
                if "Place of Issue" not in fields and len(location_candidates) >= 2:
                    fields["Place of Issue"] = location_candidates[1]
                    field_sources["Place of Issue"] = "Based on position after District of Birth"
        except ValueError:
            pass

    # Nationality detection
    full_text = " ".join(cleaned_text)
    if any(keyword in full_text for keyword in ["JAMHURI YA KENYA", "REPUBLIC OF KENYA", "KENYA"]):
        fields["Nationality"] = "Kenyan"
        field_sources["Nationality"] = "Detected 'KENYA' in OCR text"

    logging.info(f"Final extracted Kenyan ID fields: {fields}")
    logging.info(f"Field sources: {field_sources}")

    return fields

def correct_date_format(text):
    if not isinstance(text, str):
        return None  

    # Remove spaces and replace commas with dots
    cleaned = text.replace(' ', '').replace(',', '.')
    match = re.match(r'(\d{2})\.(\d{2})\.(\d{4})', cleaned)
    if match:
        day, month, year = match.groups()
        return f"{day}.{month}.{year}"
    return None


def parse_date_safe(date_str):
    try:
        return datetime.strptime(date_str, "%d.%m.%Y")
    except Exception:
        return None

def assign_dob_expiry_from_dates(dates):
    today = datetime.today()
    parsed_dates = sorted([parse_date_safe(d) for d in dates if parse_date_safe(d)])
    dob = None
    expiry = None

    # Assign DOB: earliest date <= today
    for d in parsed_dates:
        if d <= today:
            dob = d
            break

    # Assign expiry: latest date > dob (if any)
    for d in reversed(parsed_dates):
        if dob and d > dob:
            expiry = d
            break
        elif not dob:
            expiry = d
            break

    # Fallbacks if dob not assigned
    if not dob and parsed_dates:
        dob = parsed_dates[0]

    return dob.strftime("%d-%m-%Y") if dob else None, expiry.strftime("%d-%m-%Y") if expiry else None


def extract_maisha_card_fields(ocr_text):
    fields = {}

    # Step 1: Normalize text
    cleaned_text = [str(word).strip().upper() for word in ocr_text if str(word).strip()]
    if cleaned_text and cleaned_text[0] in {"JAMHURIYAKENYA", "REPUBLICOFKENYA"}:
        cleaned_text = cleaned_text[1:]

    logging.info(f"Cleaned OCR Text for Maisha Card: {cleaned_text}")

    # Step 2: Fix dates and collect them
    corrected_dates = [correct_date_format(word) for word in cleaned_text]
    dates = [d for d in corrected_dates if isinstance(d, str) and re.fullmatch(r'\d{2}\.\d{2}\.\d{4}', d)]

    # Step 3: Extract gender
    gender = next((word for word in cleaned_text if word in {"MALE", "FEMALE"}), None)
    if gender:
        fields["Gender"] = gender

    # Step 4: Extract ID number
    id_number = next((word for word in cleaned_text if re.fullmatch(r'\d{8,10}', word)), None)
    if id_number:
        fields["ID Number"] = id_number

    # Step 5: DOB from nationality
    dob_from_nationality = None
    for i, word in enumerate(cleaned_text):
        if word in {"KEN", "KENYAN"} and i + 1 < len(cleaned_text):
            possible_dob = correct_date_format(cleaned_text[i + 1])
            if parse_date_safe(possible_dob):
                dob_from_nationality = possible_dob
                fields["Nationality"] = "Kenyan"
                break

    # Step 6: Assign DOB and Expiry
    dob, expiry = assign_dob_expiry_from_dates(dates)
    if dob_from_nationality:
        nat_dob_dt = parse_date_safe(dob_from_nationality)
        dob_dt = parse_date_safe(dob) if dob else None
        if not dob_dt or (nat_dob_dt and nat_dob_dt < dob_dt):
            dob = dob_from_nationality

    if dob:
        fields["Date of Birth"] = dob
    if expiry:
        fields["Expiry Date"] = expiry

    # Step 7: Extract full names (more accurate logic)
    EXCLUDED_TOKENS = {
        "NATIONALIDENTITYCARD", "KEN", "KENYAN", "MALE", "FEMALE",
        "JAMHURIYAKENYA", "REPUBLICOFKENYA"
    }
    if gender and gender in cleaned_text:
        gender_index = cleaned_text.index(gender)
        # Grab all text before the gender keyword, filter out non-name tokens
        name_parts = [
            token for token in cleaned_text[:gender_index]
            if token.upper() not in EXCLUDED_TOKENS and any(char.isalpha() for char in token)
        ]
        if name_parts:
            full_name = " ".join(name_parts).title()
            fields["Full Names"] = full_name


    # Step 8: Locate DOB and Expiry indices
    dob_index = -1
    expiry_index = -1
    if dob:
        for i, word in enumerate(cleaned_text):
            if correct_date_format(word) == dob:
                dob_index = i
                break
    if expiry:
        for i, word in enumerate(cleaned_text):
            if correct_date_format(word) == expiry:
                expiry_index = i
                break

    logging.info(f"DOB index: {dob_index}, Expiry index: {expiry_index}")

    # Step 9: Extract location candidates
    used_tokens = set()
    if "Full Names" in fields:
        used_tokens.update(word.upper() for word in fields["Full Names"].split())
    if "Gender" in fields:
        used_tokens.add(fields["Gender"])
    if "ID Number" in fields:
        used_tokens.add(fields["ID Number"])

    location_candidates = [
        (idx, word.title())
        for idx, word in enumerate(cleaned_text)
        if word.isalpha() and word not in EXCLUDED_TOKENS and word not in used_tokens
    ]
    logging.info(f"Location candidates: {location_candidates}")

    # Step 10: Assign Place of Birth and Issue
    place_of_birth = None
    place_of_issue = None
    for idx, place in location_candidates:
        if dob_index != -1 and expiry_index != -1:
            if dob_index < idx < expiry_index and not place_of_birth:
                place_of_birth = place
            elif idx > expiry_index and not place_of_issue:
                place_of_issue = place
        elif dob_index != -1 and expiry_index == -1:
            if idx > dob_index and not place_of_birth:
                place_of_birth = place
            elif idx > dob_index and place_of_birth and not place_of_issue:
                place_of_issue = place
        elif expiry_index != -1 and dob_index == -1:
            if idx < expiry_index and not place_of_birth:
                place_of_birth = place
            elif idx > expiry_index and not place_of_issue:
                place_of_issue = place

    if place_of_birth:
        fields["Place of Birth"] = place_of_birth
    if place_of_issue and place_of_issue != place_of_birth:
        fields["Place of Issue"] = place_of_issue

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
    

def clean_text(text: str) -> str:
    """
    Cleans up OCR text:
    - Fix known OCR errors
    - Normalize spaces
    """
    text = text.replace("Aszu", "Issue")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_mrz_date(date_str: str, is_expiry: bool = False) -> str:
    """
    Parses a date in YYMMDD format from the MRZ and returns DD/MM/YYYY format.
    For expiry dates, always assume 2000+ as expiry dates are in the future.
    """
    if not date_str or len(date_str) != 6:
        return ""
    try:
        year = int(date_str[0:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])

        if is_expiry:
            full_year = 2000 + year
        else:
            current_year = datetime.now().year % 100
            century = 1900 if year > current_year else 2000
            full_year = century + year

        return f"{day:02d}/{month:02d}/{full_year:04d}"
    except Exception:
        return ""


def standardize_date(date_str: str) -> str:
    """
    Converts a date like '19 Jan 1984' into '19/01/1984'
    """
    try:
        dt = datetime.strptime(date_str, "%d %b %Y")
        return dt.strftime("%d/%m/%Y")
    except Exception:
        return date_str



def extract_text_from_passport_image(image_array: np.ndarray) -> str:
    results = reader.readtext(image_array, detail=0)
    return " ".join(results)


def extract_passport_fields(text: str) -> dict:
    text = clean_text(text)

    data = {
        "passport_number": "",
        "surname": "",
        "given_names": "",
        "nationality": "",
        "date_of_birth": "",
        "gender": "",
        "place_of_birth": "",
        "date_of_issue": "",
        "expiry_date": "",
        "raw_text": text
    }

    # --- MRZ extraction ---
    mrz_lines = re.findall(r"[A-Z0-9<]{40,}", text)
    if len(mrz_lines) >= 2:
        line1 = mrz_lines[0].ljust(44, "<")
        line2 = mrz_lines[1].ljust(44, "<")

        surname_raw = line1[5:44].split("<<")[0]
        given_names_section = line1[5:44].split("<<")[1] if "<<" in line1[5:44] else ""

        surname = surname_raw.replace("<", "").strip()

        # Parse and clean given names
        given_names_parts = given_names_section.split("<")
        given_names_clean = []
        for w in given_names_parts:
            w = w.strip()
            if not w:
                continue
            # Remove initial S if followed by all uppercase letters (e.g. SGRACE -> GRACE)
            if len(w) > 1 and w[0] == "S" and w[1:].isupper():
                w = w[1:]
            given_names_clean.append(w)

        given_names = " ".join(given_names_clean)

        passport_number = line2[0:9].replace("<", "").strip()
        nationality = line2[10:13].replace("<", "").strip()
        dob_raw = line2[13:19]
        gender = line2[20]
        expiry_raw = line2[21:27]

        data["surname"] = surname
        data["given_names"] = given_names
        data["passport_number"] = passport_number
        data["nationality"] = nationality
        data["gender"] = gender

        if dob_raw:
            data["date_of_birth"] = parse_mrz_date(dob_raw)
        if expiry_raw:
            data["expiry_date"] = parse_mrz_date(expiry_raw, is_expiry=True)

    # --- Place of birth ---
    pob_match = re.search(r"Place of Birth\s*[:\-]?\s*([A-Za-z ,.]*)", text, re.IGNORECASE)
    if pob_match:
        data["place_of_birth"] = pob_match.group(1).strip()
    else:
        pob_match = re.search(r"NAIROBI\s*[,\.]?\s*KEN", text, re.IGNORECASE)
        if pob_match:
            data["place_of_birth"] = "NAIROBI, KEN"

    # --- Other dates from text ---
    all_dates = re.findall(r"\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}", text)
    for date_str in all_dates:
        before_text = text.lower().split(date_str.lower())[0][-20:]
        if "birth" in before_text and not data["date_of_birth"]:
            data["date_of_birth"] = date_str
        elif any(word in before_text for word in ["issue", "aszu"]) and not data["date_of_issue"]:
            data["date_of_issue"] = date_str
        elif "expir" in before_text and not data["expiry_date"]:
            data["expiry_date"] = date_str

    # --- Standardize date formats ---
    for date_field in ["date_of_birth", "expiry_date", "date_of_issue"]:
        if data[date_field]:
            data[date_field] = standardize_date(data[date_field])

    return data

def process_passport_image(image_bytes: bytes) -> str:
    try:
        image = byte_to_image(image_bytes)
        processed_image = preprocess_image(image)
        np_image = np.array(processed_image)

        ocr_text = extract_text_from_passport_image(np_image)
        logging.debug(f"Extracted text: {ocr_text}")

        fields = extract_passport_fields(ocr_text)
        logging.debug(f"Final extracted Passport fields: {fields}")

        return json.dumps(fields, indent=2)
    except Exception as e:
        logging.error(f"Error processing passport image: {str(e)}")
        return json.dumps({"error": str(e)})