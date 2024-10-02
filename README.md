# Project Title

## Description
This is a project that demonstrates how to run Python code.

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
from google.api_core.exceptions import ResourceExhausted```

os: For file and environment variable operations.
whisper: To transcribe audio files.
google.generativeai: To use Google's Generative AI models.
pandas: For reading and updating CSV files.
chardet: For detecting file encodings.
time: For handling delays in case of API retries.
shutil: To move files after processing.
ResourceExhausted: Handles exceptions for quota limits from the Google API.

