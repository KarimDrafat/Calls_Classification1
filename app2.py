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

def detect_encoding(file_path):
    """Detect the encoding of the file."""
    with open(file_path, 'rb') as f:
        rawdata = f.read(10000)
        result = chardet.detect(rawdata)
        encoding = result['encoding']
        print(f"Detected encoding: {encoding}")
        return encoding

def transcribe_audio(mp3_file):
    """Transcribe the MP3 file to text using Whisper."""
    print(f"Transcribing audio file: {mp3_file}")
    
    # Check if the MP3 file exists
    if not os.path.exists(mp3_file):
        print(f"Error: The file {mp3_file} does not exist.")
        return None
    
    # Whisper's model loading
    try:
        model = whisper.load_model("tiny")  # Choose model size (tiny, base, small, etc.)
        result = model.transcribe(mp3_file)
        return result["text"]
    except FileNotFoundError as e:
        print(f"FileNotFoundError during transcription: {e}")
        return None
    except Exception as e:
        print(f"General error transcribing file {mp3_file}: {e}")
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

def process_mp3_files_and_classify(csv_file, mp3_folder):
    """Process each MP3 file, classify it, and update the CSV accordingly."""
    # Detect file encoding
    encoding = detect_encoding(csv_file)

    # Load the CSV file
    try:
        df = pd.read_csv(csv_file, encoding=encoding)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Ensure 'Call ID', 'Call Transcript', and 'Classification' columns exist
    if 'Call ID' not in df.columns:
        df['Call ID'] = ""
    if 'Call Transcript' not in df.columns:
        df['Call Transcript'] = ""
    if 'Classification' not in df.columns:
        df['Classification'] = ""

    # List MP3 files in the directory
    mp3_files = [f for f in os.listdir(mp3_folder) if f.endswith('.mp3')]

    # Print out the MP3 files detected
    print(f"MP3 files found: {mp3_files}")

    if not mp3_files:
        print(f"No MP3 files found in the folder: {mp3_folder}")
        return

    # Process each MP3 file
    for mp3_file in mp3_files:
        # Extract Call ID from the file name (e.g., "12345.mp3" -> "12345")
        call_id = os.path.splitext(mp3_file)[0]

        print(f"Processing Call ID: {call_id}")
        mp3_file_path = os.path.join(mp3_folder, mp3_file)

        # Ensure proper path formatting and print it
        mp3_file_path = mp3_file_path.replace("/", "\\")
        print(f"Full MP3 file path: {mp3_file_path}")

        try:
            # Step 1: Transcribe the audio file
            transcript = transcribe_audio(mp3_file_path)
            
            if transcript is None:
                print(f"Skipping Call ID: {call_id} due to transcription failure.")
                continue

            # Step 2: Classify the transcript
            classification = classify_transcript(transcript)

            # Step 3: Check if Call ID exists in the CSV, otherwise add a new row
            if call_id in df['Call ID'].values:
                # If Call ID exists, update the row
                df.loc[df['Call ID'] == call_id, 'Call Transcript'] = transcript
                df.loc[df['Call ID'] == call_id, 'Classification'] = classification
            else:
                # If Call ID doesn't exist, add a new row
                new_row = {
                    'Call ID': call_id,
                    'Call Transcript': transcript,
                    'Classification': classification
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            # Move the processed file to "processed_calls"
            processed_folder = os.path.join(mp3_folder, "processed_calls")
            if not os.path.exists(processed_folder):
                os.makedirs(processed_folder)
            os.rename(mp3_file_path, os.path.join(processed_folder, mp3_file))

        except Exception as e:
            print(f"Error processing Call ID {call_id}: {e}")
            continue

    # Step 4: Save the updated CSV file after all MP3 files are processed
    try:
        df.to_csv(csv_file, index=False)  # Save the updates back to the same CSV file
        print(f"File saved to {csv_file}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

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

# To run the FastAPI app, use the command: uvicorn <this_file_name>:app --reload
