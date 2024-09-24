from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import pandas as pd
import whisper
import shutil

app = FastAPI()

# Function to transcribe audio using Whisper
def transcribe_audio(mp3_file_path):
    model = whisper.load_model("tiny")
    result = model.transcribe(mp3_file_path)
    return result["text"]

# Function to classify transcripts (dummy function for demonstration)
def classify_transcript(transcript):
    # In a real-world case, replace this with your classification logic
    return "Positive" if "good" in transcript else "Negative"

# Function to process MP3 files and update the CSV
def process_files_and_update_csv(mp3_files, csv_file):
    # Load or create CSV file
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["Call ID", "Call Transcript", "Classification"])
    
    results = []
    
    for mp3_file in mp3_files:
        call_id = os.path.splitext(mp3_file.filename)[0]
        file_path = f"uploads/{mp3_file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(mp3_file.file, buffer)
        
        # Transcribe audio
        transcript = transcribe_audio(file_path)
        
        # Classify transcript
        classification = classify_transcript(transcript)
        
        # Update CSV
        new_row = {"Call ID": call_id, "Call Transcript": transcript, "Classification": classification}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Add result to return to client
        results.append({"call_id": call_id, "transcript": transcript, "classification": classification})
        
        # Remove the processed file
        os.remove(file_path)
    
    # Save updated CSV
    df.to_csv(csv_file, index=False)
    
    return results

# Endpoint to handle file uploads and processing
@app.post("/upload")
async def upload_mp3_files(mp3_files: list[UploadFile] = File(...)):
    if not mp3_files:
        raise HTTPException(status_code=400, detail="No MP3 files uploaded.")
    
    # Process MP3 files and update CSV
    csv_file = "static/call_data.csv"
    results = process_files_and_update_csv(mp3_files, csv_file)
    
    return JSONResponse(content={"results": results})
