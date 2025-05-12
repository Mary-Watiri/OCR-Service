from fastapi import FastAPI, File, UploadFile
from ocr_utils import preprocess_image, extract_text_from_image, extract_fields
import logging
import os

app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

@app.post("/extract-fields/")
async def extract_fields_endpoint(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_file_path = "temp_image.jpg"
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)
        logging.info(f"File '{file.filename}' saved for processing.")

        # OCR processing
        processed_image = preprocess_image(temp_file_path)
        ocr_results = extract_text_from_image(processed_image)
        fields = extract_fields(ocr_results)

        if not isinstance(fields, dict):
            logging.error(f"Expected a dictionary from extract_fields, got {type(fields)}")
            fields = {}

        return {"extracted_fields": fields}
    except Exception as e:
        logging.exception(f"Failed to extract fields: {str(e)}")
        return {"error": "Field extraction failed", "details": str(e)}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logging.info("Temporary file deleted.")

@app.get("/")
async def root():
    return {"message": "OCR microservice is running"}
