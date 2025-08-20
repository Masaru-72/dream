import json
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse # <-- IMPORT FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv() 

from dream_rag import interpret_dream_rag
from image_generator import generate_image_async

class DreamDetail(BaseModel):
    question: str
    answer: str

class DreamRequest(BaseModel):
    dream: str
    feeling: str
    details: List[DreamDetail]

class ImageRequest(BaseModel):
    prompt: str


app = FastAPI(title="Dream Interpreter API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main index.html file."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws/interpret")
async def interpret_dream_ws(websocket: WebSocket):
    """The WebSocket endpoint that connects clients to the RAG engine."""
    await websocket.accept()
    try:
        json_str = await websocket.receive_text()
        data = json.loads(json_str)
        
        try:
            request_data = DreamRequest(**data)
        except Exception as e:
            print(f"Data validation error: {e}")
            await websocket.send_json({"error": "Invalid data format."})
            return

        details_as_dicts = [detail.model_dump() for detail in request_data.details]

        json_response_str = await interpret_dream_rag(
            dream_text=request_data.dream,
            feeling=request_data.feeling,
            details=details_as_dicts 
        )

        await websocket.send_text(json_response_str)

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred in the WebSocket endpoint: {e}")
    finally:
        await websocket.close()
        print("WebSocket connection closed by server.")


@app.post("/api/generate-image")
async def create_dream_image(request: ImageRequest):
    """
    Receives a prompt, generates an image, and returns the image
    file directly in the response body.
    """
    try:
        print(f"Received request to generate image for prompt: '{request.prompt[:70]}...'")
        image_path = await generate_image_async(request.prompt)
        
        if not image_path or "error.png" in image_path:
             raise Exception("Image generation failed.")
        return FileResponse(image_path, media_type="image/png")

    except Exception as e:
        print(f"Failed to generate or return image file: {e}")
        return FileResponse("static/images/tarot.jpg", media_type="image/jpg", status_code=500)