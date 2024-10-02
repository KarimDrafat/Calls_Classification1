from flask import Flask, request, render_template, jsonify
import os
import whisper
import google.generativeai as genai
import pandas as pd
import time
from google.api_core.exceptions import ResourceExhausted

# Initialize Flask app
app = Flask(__name__)

# Configure the API key for the Generative AI service
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 0,
    "top_p": 0,
    "top_k": 64,
    "max_output_tokens": 10092,
    "response_mime_type": "text/plain",
}

# Fetch the instruction from the environment variable
instruction = os.getenv("instructions")

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=instruction,  # Use the instruction from the .env file
)

def transcribe_audio(mp3_file):
    """Transcribe the MP3 file to text using Whisper."""
    model = whisper.load_model("tiny")
    result = model.transcribe(mp3_file)
    return result["text"]

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
            time.sleep(wait_time)
        except Exception as e:
            return "Error"

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/transcribe', methods=['POST'])
def handle_transcription():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.mp3'):
        # Save the uploaded file
        mp3_file_path = os.path.join('uploads', file.filename)
        file.save(mp3_file_path)

        # Step 1: Transcribe the audio file
        transcript = transcribe_audio(mp3_file_path)
        
        # Step 2: Classify the transcript
        classification = classify_transcript(transcript)

        # Step 3: Return the results as a JSON object
        result = {
            "transcript": transcript,
            "classification": classification
        }

        # Cleanup: Optionally delete the uploaded file after processing
        os.remove(mp3_file_path)

        return jsonify(result)

    return jsonify({"error": "Invalid file format, please upload a .mp3 file"}), 400

if __name__ == '__main__':
    # Ensure uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
