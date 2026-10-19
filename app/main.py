"""FastAPI app: upload an image, get a prediction and see what registers change.

Start it (from the repo root):
    uvicorn app.main:app --port 8000
Then open http://localhost:8000 in a browser, or call the API directly:
    curl -F "file=@my_photo.jpg" http://localhost:8000/predict

Endpoints:
    GET  /          web page with an upload form
    POST /predict   multipart form with one image file called "file"; returns JSON
    GET  /health    whether the models are loaded
    GET  /docs      interactive API documentation (made by FastAPI)
"""

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app.inference import load_all, predict

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MODELS: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models once when the server starts (the first start downloads DINOv2)."""
    start = time.time()
    MODELS.update(load_all())
    print(f"Models loaded in {time.time() - start:.0f}s")
    yield
    MODELS.clear()


app = FastAPI(
    title="Vision Transformers Need Registers: demo",
    description="Upload an image. Our ViT-Tiny classifiers (0 and 4 registers) predict its class, and "
                "DINOv2 without and with registers show their attention and token-norm maps.",
    version="1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if MODELS else "loading",
        "classifiers": [c.name for c in MODELS.get("classifiers", [])],
        "explainers": [e.name for e in MODELS.get("explainers", [])],
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(..., description="An image file (JPEG, PNG, ...)")) -> dict:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The file is not an image that can be opened.")

    start = time.time()
    result = predict(MODELS, image)
    result["filename"] = file.filename
    result["seconds"] = round(time.time() - start, 2)
    return result
