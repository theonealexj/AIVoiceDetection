import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    print("Error: API_KEY not found in .env file.")
    exit(1)

print(f"Using API Key: {API_KEY}")

# Read the sample base64 audio
try:
    with open("sample_base64.txt", "r") as f:
        audio_data = f.read().strip()
except FileNotFoundError:
    print("Error: sample_base64.txt not found.")
    exit(1)

# API Endpoint
url = "http://localhost:8000/detect-voice"

# Headers with API Key
headers = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY
}

# Payload
payload = {
    "audio_base64": audio_data,
    "Language": "English",
    "Audio Format": "wav"
}

print(f"Sending request to {url}...")
try:
    response = requests.post(url, headers=headers, json=payload)
    
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))

except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the server. Make sure it is running (run_server.bat).")
except Exception as e:
    print(f"An error occurred: {e}")
