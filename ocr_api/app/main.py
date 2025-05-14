from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ocr_api.app.ocr_utils import process_id_image, process_maisha_card_image
import shutil
import os
import uuid

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/process/id")
async def process_id(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = process_id_image(file_path)
    os.remove(file_path)
    return JSONResponse(content=result)

@app.post("/process/maisha")
async def process_maisha(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = process_maisha_card_image(file_path)
    os.remove(file_path)
    return JSONResponse(content=result)

@app.get("/")
def read_root():
    return {"message": "OCR Service is running. Use /process/id or /process/maisha"}
