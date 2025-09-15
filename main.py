from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, StreamingResponse
import logging
import json
import io
import cv2
import base64
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- Your utils ---
from ocr_utils import process_id_image, process_maisha_card_image, process_passport_image
from signature_utils import extract_signature

# --- FastAPI App ---
app = FastAPI()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=8) 

def run_in_threadpool(func, *args):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(executor, func, *args)


# --- Utility function ---
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


# --- Endpoints ---

@app.post("/process/id")
async def process_id(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result_json_str = await run_in_threadpool(process_id_image, image_bytes)
        result_dict = json.loads(result_json_str)
        return JSONResponse(content=result_dict)
    except Exception as e:
        logger.error(f"Error processing ID image: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/process/maisha")
async def process_maisha(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result_json_str = await run_in_threadpool(process_maisha_card_image, image_bytes)
        result_dict = json.loads(result_json_str)
        return JSONResponse(content=result_dict)
    except Exception as e:
        logger.error(f"Error processing Maisha card image: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/process/passport")
async def process_passport(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result_json_str = await run_in_threadpool(process_passport_image, image_bytes)
        result_dict = json.loads(result_json_str)
        return JSONResponse(content=result_dict)
    except Exception as e:
        logger.error(f"Error processing Passport image: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/signature/detect")
async def detect_signature(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        signature_img, confidence, status = await run_in_threadpool(extract_signature, image_bytes)

        if signature_img is None:
            return JSONResponse({
                "signature_detected": False,
                "confidence": float(confidence),
                "message": status
            })

        # Resize & pad
        signature_img = resize_and_pad_image(signature_img)
        success, buffer = cv2.imencode(".png", signature_img)
        if not success:
            return JSONResponse(status_code=500, content={"error": "Encoding PNG failed"})

        # Always return JSON response with base64 image
        b64_img = base64.b64encode(buffer).decode("utf-8")
        return JSONResponse({
            "signature_detected": True,
            "confidence": float(confidence),
            "message": status,
            "signature_base64": b64_img,
            "signature_format": "image/png"
        })
        
    except Exception as e:
        logger.error(f"Error detecting signature: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/signature/detect/image")
async def detect_signature_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    signature_img, confidence, status = await run_in_threadpool(extract_signature, image_bytes)

    if signature_img is None:
        return JSONResponse({
            "signature_detected": False,
            "confidence": float(confidence),
            "message": status
        })

    # Resize & pad
    signature_img = resize_and_pad_image(signature_img)
    success, buffer = cv2.imencode(".png", signature_img)
    if not success:
        return JSONResponse(status_code=500, content={"error": "Encoding PNG failed"})

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

@app.get("/")
def read_root():
    return {
        "message": "OCR Service is running. Use /process/id, /process/maisha, /process/passport, or /signature/detect"
    }
