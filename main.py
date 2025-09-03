from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ocr_utils import process_id_image, process_maisha_card_image, process_passport_image
import logging
from passporteye import read_mrz
import os
import tempfile
import shutil
from fastapi.responses import StreamingResponse
import io
import cv2
import base64
import numpy as np
from fastapi import Query
from signature_utils import extract_signature


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
    
def resize_and_pad_image(rgba, min_size=150, max_size=500):
    h, w = rgba.shape[:2]
    target_size = max(min(max(h, w), max_size), min_size)
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded = np.zeros((target_size, target_size, 4), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return padded

@app.post("/signature/detect")
async def detect_signature(file: UploadFile = File(...), as_json: bool = Query(False)):
    image_bytes = await file.read()
    signature_img, confidence, status = extract_signature(image_bytes)

    if signature_img is None:
        return JSONResponse({
            "signature_detected": False,
            "confidence": confidence,
            "message": status
        })

    success, buffer = cv2.imencode(".png", signature_img)
    if not success:
        return JSONResponse(status_code=500, content={"error": "Encoding PNG failed"})

    if as_json:
        b64_img = base64.b64encode(buffer).decode("utf-8")
        return JSONResponse({
            "signature_detected": True,
            "confidence": confidence,
            "message": status,
            "signature_base64": b64_img
        })

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/png",
        headers={
            "X-Signature-Detected": "true",
            "X-Confidence": str(confidence),
            "X-Status": status,
            "Content-Disposition": "inline; filename=signature.png"
        }
    )

@app.get("/")
def read_root():
    return {"message": "OCR Service is running. Use /process/id or /process/maisha or /process/passport"}
