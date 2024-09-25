import os
import whisper 
import google.generativeai as genai
import pandas as pd
import chardet
import time
import shutil  # For moving files
from google.api_core.exceptions import ResourceExhausted
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Configure the API key for the Generative AI service
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create the classification model
generation_config = {
    "temperature": 0,
    "top_p": 0,
    "top_k": 64,
    "max_output_tokens": 10092,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=("instructions"),
)

# Create FastAPI app
app = FastAPI()

# Mount the static directory for serving HTML and other static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    """Serve the main HTML file."""
    return FileResponse("static/index.html")  # Path to your HTML file

# Your other logic and endpoints go here
# e.g., detect_encoding, transcribe_audio, classify_transcript, process_mp3_files_and_classify, etc.

@app.post("/process")
async def process_files(csv_file_name: str, mp3_folder_name: str):
    """Endpoint to process MP3 files and classify them."""
    csv_file_path = os.path.join(os.getcwd(), csv_file_name)
    mp3_folder_path = os.path.join(os.getcwd(), mp3_folder_name)

    # Check if CSV and MP3 folder exist
    if not os.path.exists(csv_file_path):
        raise HTTPException(status_code=404, detail=f"The CSV file {csv_file_path} does not exist.")
    if not os.path.exists(mp3_folder_path):
        raise HTTPException(status_code=404, detail=f"The MP3 folder {mp3_folder_path} does not exist.")

    # Process the MP3 files and update the CSV
    process_mp3_files_and_classify(csv_file_path, mp3_folder_path)
    return {"message": "Processing complete!"}
