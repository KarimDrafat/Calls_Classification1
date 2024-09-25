import os
import whisper 
import google.generativeai as genai
import pandas as pd
import chardet
import time
import shutil  # For moving files
from google.api_core.exceptions import ResourceExhausted
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List

# Configure the API key for the Generative AI service
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create the classification model
generation_config = {
    "temperature": 0,
    "top_p": 0,
    "top_k": 64,
    "max_output_tokens": 10092,  # Adjusted token limit
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

def transcribe_audio(mp3_file_path):
    """Transcribe the MP3 file to text using Whisper."""
    print(f"Transcribing audio file: {mp3_file_path}")
    
    # Whisper's model loading
    try:
        model = whisper.load_model("tiny")  # Choose model size (tiny, base, small, etc.)
        result = model.transcribe(mp3_file_path)
        return result["text"]
    except FileNotFoundError as e:
        print(f"FileNotFoundError during transcription: {e}")
        return None
    except Exception as e:
        print(f"General error transcribing file {mp3_file_path}: {e}")
        return None

def classify_transcript(transcript):
    """Classify the call transcript using Generative AI."""
    chat_session = model.start_chat()
    retry_count = 0

    while retry_count < 5:
        try:
            response = chat_session.send_message(transcript)
            return response.text
        except ResourceExhausted:
            retry_count += 1
            wait_time = 2 ** retry_count  # Exponential backoff
            print(f"Quota exceeded, retrying after {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error classifying transcript: {e}")
            return "Error"

@app.post("/process")
async def process_files(mp3_files: List[UploadFile] = File(...)):
    """Endpoint to process MP3 files and classify them."""
    results = []
    
    for mp3_file in mp3_files:
        # Save the uploaded file to a temporary location
        mp3_file_path = f"temp_{mp3_file.filename}"
        with open(mp3_file_path, "wb") as f:
            shutil.copyfileobj(mp3_file.file, f)
        
        # Step 1: Transcribe the audio file
        transcript = transcribe_audio(mp3_file_path)
        
        if transcript is None:
            print(f"Skipping file {mp3_file.filename} due to transcription failure.")
            continue

        # Step 2: Classify the transcript
        classification = classify_transcript(transcript)

        # Add results to the response
        results.append({
            "call_id": mp3_file.filename,
            "transcript": transcript,
            "classification": classification
        })

        # Remove the temporary MP3 file
        os.remove(mp3_file_path)

    return {"results": results}

# To run the FastAPI app, use the command: uvicorn <this_file_name>:app --reload
