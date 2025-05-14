import cv2
import easyocr
import numpy as np
import re
import json
import logging
from typing import List, Dict


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Initialize EasyOCR
reader = easyocr.Reader(['en', 'sw'], gpu=False)

def preprocess_image(image_path, resize_factor=2.0, light_text=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    h, w = image.shape[:2]
    if max(h, w) < 500:
        resize_factor = min(1.5, resize_factor)
    image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(denoised)

    thresh_type = cv2.THRESH_BINARY_INV if light_text else cv2.THRESH_BINARY
    thresh = cv2.adaptiveThreshold(
        contrasted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, 15, 11
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    processed = cv2.bitwise_not(closed) if light_text else closed
    return processed

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
    # Match common date formats like DD/MM/YYYY, DD-MM-YYYY, MM/DD/YYYY, etc.
    return bool(re.match(r'^\d{2}[./-]\d{2}[./-]\d{4}$', text))

def is_word(word: str) -> bool:
    # Check if the word consists only of alphabetic characters and is non-empty
    return bool(re.match(r"^[A-Za-z]+$", word))

def is_id_number(word: str) -> bool:
    # Assuming ID Number follows a specific pattern like a numeric string of 12 digits
    return bool(re.match(r"^\d{12}$", word))  # Adjust the regex based on the actual format


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

def process_id_image(image_path):
    try:
        processed_image = preprocess_image(image_path, light_text=False)
        ocr_text = extract_text_from_image(processed_image)
        fields = extract_fields(ocr_text)
        return json.dumps(fields, indent=2)
    except Exception as e:
        logging.error(f"Error processing ID image: {str(e)}")
        return json.dumps({"error": str(e)})

def process_maisha_card_image(image_path):
    try:
        processed_image = preprocess_image(image_path, resize_factor=3.2, light_text=True)
        ocr_text = extract_text_from_image(processed_image)
        
        logging.info(f"Extracted text: {ocr_text}")
        
        fields = extract_maisha_card_fields(ocr_text)
        
        logging.info(f"Final extracted Maisha Card fields: {fields}")
        
        return json.dumps(fields, indent=2)
    except Exception as e:
        logging.error(f"Error processing Maisha card image: {str(e)}")
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    # Test processing an ID image
    id_image_path = "kenyaNationalID-circle-2900793588.png"
    result_id = process_id_image(id_image_path)
    print("ID Image Result:", result_id)

    # Test processing a Maisha card image
    maisha_image_path = "WhatsApp Image 2025-05-06 at 3.45.07 PM(1).jpeg"
    result_maisha = process_maisha_card_image(maisha_image_path)
    print("Maisha Card Image Result:", result_maisha)