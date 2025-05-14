import json
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ocr_api.app.ocr_utils import process_id_image, process_maisha_card_image

app = FastAPI()

@app.post("/process/id")
async def process_id(file: UploadFile = File(...)):
    try:
        # Read the uploaded file as bytes
        image_bytes = await file.read()
        
        # Process the ID image using the utility function
        result = process_id_image(image_bytes)
        
        # Return the processed result as a JSON response
        return JSONResponse(content=json.loads(result))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error processing ID image: {str(e)}"})

@app.post("/process/maisha")
async def process_maisha(file: UploadFile = File(...)):
    try:
        # Read the uploaded file as bytes
        image_bytes = await file.read()
        
        # Process the Maisha card image using the utility function
        result = process_maisha_card_image(image_bytes)
        
        # Return the processed result as a JSON response
        return JSONResponse(content=json.loads(result))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error processing Maisha card image: {str(e)}"})

@app.get("/")
def read_root():
    return {"message": "OCR Service is running. Use /process/id or /process/maisha"}

