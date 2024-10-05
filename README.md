# Automatic Speech-to-Text and Text Classification System Using LLM API

## Description
This project implements an Audio Transcription and Classification System that automates the conversion of audio calls into text and categorizes the content using advanced AI technologies. By integrating Whisper for transcription and API from Google Generative AI for classification , the system enhances operational efficiency within organizations that rely on audio data, such as customer service and sales departments.

The system works by detecting the encoding of audio files, transcribing them to text, and classifying the content based on predefined criteria. It updates a CSV file to maintain accurate records of call transcripts and classifications, ensuring compliance with industry standards. The automation not only improves the speed and accuracy of transcription but also provides valuable insights into customer interactions, allowing for data-driven decision-making.

Key benefits of the project include:

Cost Reduction: Decreases the cost per minute of audio/calls processing, significantly lowering departmental expenses.

Improved Quality Assurance: Enables systematic monitoring of interactions to enhance employee performance and customer satisfaction.

Real-time Insights: Provides immediate feedback for prompt issue resolution and strategic adjustments.

Scalability: Adapts to increasing volumes of audio data without proportional workforce increases.

Customizable Analysis: Allows for tailored classification to align with specific business goals, refining marketing and service delivery efforts.

Overall, this project serves as a powerful tool for organizations looking to leverage audio data more effectively, enhance customer service, and drive continuous improvement.

## Example Code
#### Step 1: Import Required Libraries

The script begins by importing necessary libraries and modules. These handle file operations, transcription, AI model configuration, and CSV processing.
```import os
import whisper 
import google.generativeai as genai
import pandas as pd
import chardet
import time
import shutil
from google.api_core.exceptions import ResourceExhausted` ``` `
```

## Step 2: Configure API Key for Google Generative AI

Set up the API key from an environment variable to access Google's Generative AI services.

```
#Configure the API key for the Generative AI service
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
```
This retrieves the API key from the environment variable and configures the Generative AI library.

## Step 3: Configure the Generation Model

Set up the configuration for the AI model used to classify transcriptions. Instruction is saved in .env file
```
generation_config = {
    "temperature": 0,
    "top_p": 0,
    "top_k": 64,
    "max_output_tokens": 10092,
    "response_mime_type": "text/plain",
}

instruction = os.getenv("instructions")
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=instruction,
)
```
This code defines the model parameters and loads the classification model using the environment variable instructions.

## Step 4: Detect Encoding of CSV File

Detects the file encoding of the CSV to ensure it is correctly read by pandas.
```
def detect_encoding(file_path):
    """Detect the encoding of the file."""
    with open(file_path, 'rb') as f:
        rawdata = f.read(10000)
        result = chardet.detect(rawdata)
        encoding = result['encoding']
        print(f"Detected encoding: {encoding}")
        return encoding
```
This function is used to identify the character encoding of the CSV file to prevent encoding errors.
## Step 5: Transcribe MP3 File to Text Using Whisper

Uses the Whisper model to transcribe an MP3 file to text.
```
def transcribe_audio(mp3_file):
    """Transcribe the MP3 file to text using Whisper."""
    if not os.path.exists(mp3_file):
        print(f"Error: The file {mp3_file} does not exist.")
        return None
    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(mp3_file)
        return result["text"]
    except FileNotFoundError as e:
        print(f"FileNotFoundError during transcription: {e}")
        return None
    except Exception as e:
        print(f"General error transcribing file {mp3_file}: {e}")
        return None
```
This function takes the MP3 file, transcribes it into text, and handles errors in case the file is missing or there's a problem with transcription.

## Step 6: Classify the Transcript Using Generative AI

This function classifies the transcribed text using the Generative AI model and handles retries in case of quota exhaustion.
```
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
```
This handles the API call to the Generative AI model, with error handling for quota limits and retries.


## Step 7: Process MP3 Files and Classify Transcripts

This function processes each MP3 file, transcribes it, classifies the transcript, and updates the CSV file.

```

def process_mp3_files_and_classify(csv_file, mp3_folder):
    """Process each MP3 file, classify it, and update the CSV accordingly."""
    encoding = detect_encoding(csv_file)
    try:
        df = pd.read_csv(csv_file, encoding=encoding)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Ensure necessary columns exist
    if 'Call ID' not in df.columns:
        df['Call ID'] = ""
    if 'Call Transcript' not in df.columns:
        df['Call Transcript'] = ""
    if 'Classification' not in df.columns:
        df['Classification'] = ""

    mp3_files = [f for f in os.listdir(mp3_folder) if f.endswith('.mp3')]
    print(f"MP3 files found: {mp3_files}")

    if not mp3_files:
        print(f"No MP3 files found in the folder: {mp3_folder}")
        return

    for mp3_file in mp3_files:
        call_id = os.path.splitext(mp3_file)[0]
        print(f"Processing Call ID: {call_id}")
        mp3_file_path = os.path.join(mp3_folder, mp3_file).replace("/", "\\")

        try:
            transcript = transcribe_audio(mp3_file_path)
            if transcript is None:
                continue
            classification = classify_transcript(transcript)

            if call_id in df['Call ID'].values:
                df.loc[df['Call ID'] == call_id, 'Call Transcript'] = transcript
                df.loc[df['Call ID'] == call_id, 'Classification'] = classification
            else:
                new_row = {'Call ID': call_id, 'Call Transcript': transcript, 'Classification': classification}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            processed_folder = os.path.join(mp3_folder, "processed_calls")
            if not os.path.exists(processed_folder):
                os.makedirs(processed_folder)
            os.rename(mp3_file_path, os.path.join(processed_folder, mp3_file))
        except Exception as e:
            print(f"Error processing Call ID {call_id}: {e}")
            continue

    try:
        df.to_csv(csv_file, index=False)
        print(f"File saved to {csv_file}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")
```
This function:
Detects the encoding of the CSV.
Reads and updates the CSV file with transcriptions and classifications.
Processes each MP3 file by transcribing and classifying them.
Moves processed MP3 files to a "processed_calls" folder.

## Step 8: Define File Paths and Start Processing

Finally, define the file paths and start processing the MP3 files.

```
csv_file_name = "Transcripts.csv"
mp3_folder = "mp3_calls"
csv_file_path = os.path.join(os.getcwd(), csv_file_name)
mp3_folder_path = os.path.join(os.getcwd(), mp3_folder)

if not os.path.exists(csv_file_path):
    print(f"Error: The CSV file {csv_file_path} does not exist.")
else:
    if not os.path.exists(mp3_folder_path):
        print(f"Error: The MP3 folder {mp3_folder_path} does not exist.")
    else:
        process_mp3_files_and_classify(csv_file_path, mp3_folder_path)
```
This code checks for the existence of the CSV and MP3 folder, and if both exist, it starts processing the files.
