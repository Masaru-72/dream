import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path
from datetime import datetime
import asyncio

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./model_cache"
# IMPORTANT: Save inside 'static' to make it web-accessible if you switch to URL method
# Also, FileResponse needs a path relative to where you run the server.
OUTPUT_DIR = Path("static/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Image Generator: Using device: {DEVICE}")

# --- Load Model (Singleton Pattern) ---
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.to(DEVICE)
print("Image Generator: Stable Diffusion model loaded.")

# --- The Core Function ---
def generate_and_save_image(prompt: str) -> str:
    """
    Generates an image from a prompt, saves it, and returns the
    full file system path to the saved image.
    """
    print(f"Generating image for prompt: '{prompt[:50]}...'")
    try:
        image = pipe(prompt).images[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize prompt for filename
        short_prompt = "".join(filter(str.isalnum, prompt.lower().split()[:4]))[:30]
        filename = f"{short_prompt}_{timestamp}.png"
        image_path = OUTPUT_DIR / filename
        
        image.save(image_path)
        print(f"Image saved to {image_path}")

        return str(image_path)
    except Exception as e:
        print(f"Error during image generation: {e}")
        return "static/images/error.png" # Return path to a placeholder

async def generate_image_async(prompt: str) -> str:
    """
    Runs the synchronous generation function in a separate thread
    to avoid blocking the main FastAPI event loop.
    """
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(
        None, generate_and_save_image, prompt
    )
    return path