import os
import whisper 
import google.generativeai as genai
import pandas as pd
import chardet
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from google.api_core.exceptions import ResourceExhausted

app = FastAPI()

# Add CORS middleware to allow requests from frontend (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    system_instruction=("You should analyze the Call Transcript column and classify it without giving any details. "
                        "Please note that each row in the column is about a conversation "
                        "between a customer and an agent where the customer is either trying "
                        "to ask a question, buy, or complain while the agent provides help. "
                        "So your job is to classify each row into the following list based on "
                        "[\"Complaint\", \"Query\", \"Compliment\", "
                        "\"late-Delivery\",\"Exchange\", \"Other\"] or if it is a combination of both, classify it with both classes as for example : Complaint/Query"),
)

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
        return []

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
        return []

    results = []  # Store the results for display

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

            results.append((call_id, transcript, classification))  # Store the result

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

    return results  # Return the results for display

@app.post("/upload/")
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload multiple MP3 files for processing."""
    csv_file_name = "fashion1.csv"
    mp3_folder = "mp3_calls"

    # Ensure the mp3 folder exists
    if not os.path.exists(mp3_folder):
        os.makedirs(mp3_folder)

    # Save uploaded files to the designated folder
    for file in files:
        if file.filename.endswith('.mp3'):
            file_location = os.path.join(mp3_folder, file.filename)
            with open(file_location, "wb") as f:
                f.write(await file.read())
        else:
            return {"error": "Only .mp3 files are allowed"}

    # Process the MP3 files and get results
    csv_file_path = os.path.join(os.getcwd(), csv_file_name)
    results = process_mp3_files_and_classify(csv_file_path, mp3_folder)

    # Prepare the HTML response with results
    result_table = """
    <html>
        <head>
            <title>Processing Results</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f4f4f4;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                }
                h1, h2 {
                    color: #333;
                }
                table {
                    width: 80%;
                    border-collapse: collapse;
                    margin-top: 20px;
                    background-color: white;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                th, td {
                    padding: 12px;
                    border: 1px solid #ddd;
                    text-align: left;
                }
                th {
                    background-color: #007BFF;
                    color: white;
                }
                tr:nth-child(even) {
                    background-color: #f2f2f2;
                }
                tr:hover {
                    background-color: #ddd;
                }
                .message {
                    margin-top: 20px;
                    font-size: 16px;
                    color: #28a745;
                }
                .back-link {
                    margin-top: 20px;
                    display: inline-block;
                    padding: 10px 15px;
                    background-color: #007BFF;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                }
                .back-link:hover {
                    background-color: #0056b3;
                }
            </style>
        </head>
        <body>
            <h1>Processing Results</h1>
            <table>
                <tr>
                    <th>Call ID</th>
                    <th>Transcript</th>
                    <th>Classification</th>
                </tr>
    """
    
    for call_id, transcript, classification in results:
        result_table += f"""
                <tr>
                    <td>{call_id}</td>
                    <td>{transcript}</td>
                    <td>{classification}</td>
                </tr>
        """
        
    result_table += """
            </table>
            <div class="message">Results have been successfully added to the CSV file.</div>
            <a class="back-link" href="/">Back to Upload</a>
        </body>
    </html>
    """
    
    return HTMLResponse(content=result_table)

@app.get("/")
async def main():
    """Render the upload form."""
    content = """
    <html>
        <head>
            <title>Upload MP3 Files</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f4f4f4;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                }
                h1 {
                    color: #333;
                }
                form {
                    margin-top: 20px;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                }
                input[type="file"] {
                    margin-bottom: 10px;
                }
                button {
                    padding: 10px 15px;
                    background-color: #007BFF;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }
                button:hover {
                    background-color: #0056b3;
                }
            </style>
        </head>
        <body>
            <h1>Upload MP3 Files</h1>
            <form action="/upload/" enctype="multipart/form-data" method="post">
                <input name="files" type="file" multiple accept=".mp3">
                <button type="submit">Process Classification</button>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

