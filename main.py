from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ocr_utils import process_id_image, process_maisha_card_image
import logging
import os

app = FastAPI()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@app.post("/process/id")
async def process_id(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = process_id_image(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing ID image: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/process/maisha")
async def process_maisha(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = process_maisha_card_image(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing Maisha card image: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def read_root():
    return {"message": "OCR Service is running. Use /process/id or /process/maisha"}
