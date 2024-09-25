import os
import whisper
import google.generativeai as genai
import pandas as pd
import chardet
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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

app = FastAPI()

# Mount the static directory for serving the HTML
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

def transcribe_audio(mp3_file_path):
    """Transcribe the MP3 file to text using Whisper."""
    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(mp3_file_path)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def classify_transcript(transcript):
    """Classify the transcript using Generative AI."""
    try:
        chat_session = model.start_chat()
        response = chat_session.send_message(transcript)
        return response.text
    except Exception as e:
        print(f"Error classifying transcript: {e}")
        return "Error"

@app.post("/process")
async def process_files(mp3_files: List[UploadFile] = File(...)):
    results = []
    for mp3_file in mp3_files:
        call_id = os.path.splitext(mp3_file.filename)[0]
        file_path = f"temp/{mp3_file.filename}"
        
        # Save uploaded MP3 file to the server
        with open(file_path, "wb") as f:
            f.write(await mp3_file.read())
        
        # Transcribe the MP3 file
        transcript = transcribe_audio(file_path)
        if not transcript:
            return HTTPException(status_code=500, detail=f"Failed to transcribe {mp3_file.filename}")

        # Classify the transcript
        classification = classify_transcript(transcript)

        # Add results to the list
        results.append({
            "call_id": call_id,
            "transcript": transcript,
            "classification": classification
        })

        # Update CSV file
        csv_file = "transcripts.csv"
        try:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
            else:
                df = pd.DataFrame(columns=["Call ID", "Transcript", "Classification"])

            df = df.append({"Call ID": call_id, "Transcript": transcript, "Classification": classification}, ignore_index=True)
            df.to_csv(csv_file, index=False)
        except Exception as e:
            return HTTPException(status_code=500, detail=f"Error updating CSV file: {e}")

    return {"results": results}
