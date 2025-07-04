from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ocr_utils import process_id_image, process_maisha_card_image, process_passport_image
import logging
from passporteye import read_mrz
import os
import tempfile
import shutil

app = FastAPI()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import json

@app.post("/process/id")
async def process_id(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result_json_str = process_id_image(image_bytes) 
        result_dict = json.loads(result_json_str)      
        return JSONResponse(content=result_dict)
    except Exception as e:
        logger.error(f"Error processing ID image: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/process/maisha")
async def process_maisha(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result_json_str = process_maisha_card_image(image_bytes) 
        result_dict = json.loads(result_json_str)
        return JSONResponse(content=result_dict)
    except Exception as e:
        logger.error(f"Error processing Maisha card image: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    

@app.post("/process/passport")
async def process_passport(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result_json_str = process_passport_image(image_bytes) 
        result_dict = json.loads(result_json_str)
        return JSONResponse(content=result_dict)
    except Exception as e:
        logger.error(f"Error processing Passport image: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.get("/")
def read_root():
    return {"message": "OCR Service is running. Use /process/id or /process/maisha or /process/passport"}
