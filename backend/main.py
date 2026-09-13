from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model_utils import load_model, predict_and_explain

MODEL_PATH = r"D:\Deepfake-detector\backend\deepfake_detector_final.keras"

app = FastAPI(title="Deepfake Detector API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, base_model = load_model(MODEL_PATH)


@app.get("/")
def root():
    return {"status": "ok", "message": "Deepfake Detector API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()
    try:
        pil_image = Image.open(BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    result = predict_and_explain(pil_image, model, base_model)
    return result