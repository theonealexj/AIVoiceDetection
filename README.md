# AI-Generated Voice Detection System

A production-ready REST API to detect AI-generated voice samples vs. Human voice samples using a Deep Learning (CNN) approach based on acoustic features.

## Project Structure
- `app/`: Contains the FastAPI application and inference logic.
- `training/`: Scrips for dataset generation and model training.
- `data/`: Storage for checking human/ai samples.
- `Dockerfile`: Deployment configuration.

## Setup & Installation

### Prerequisites
- Python 3.9+
- Docker (optional)

### Local Setup
The project works best with the included virtual environment (`.venv`).

1. **Quick Start (Windows)**:
   This command installs dependencies, trains the model, and starts the API.
   ```bash
   .\run_all.bat
   ```

2. **Run Server Only**:
   If you have already trained the model:
   ```bash
   .\run_server.bat
   ```

3. **Manual Development**:
   Use the virtual environment python explicitly:
   ```bash
   .\.venv\Scripts\python -m pip install -r requirements.txt
   .\.venv\Scripts\python training/train.py
   ```

## API Usage

### Endpoint: `/detect-voice`
- **Method**: `POST`
- **Headers**: 
  - `x-api-key: secret-key-123` OR `Authorization: Bearer secret-key-123`
- **Body**: JSON
  ```json
  {
      "audio_base64": "<BASE64_ENCODED_MP3_STRING>"
  }
  ```

### Response
```json
{
  "classification": "AI_GENERATED",
  "confidence": 0.98,
  "explainability": {
    "spectral_smoothness": 12.5,
    "pitch_variance": 0.0
  }
}
```

## Docker Deployment

1. **Build Image**:
   ```bash
   docker build -t voice-detector .
   ```

2. **Run Container**:
   ```bash
   docker run -p 8000:8000 voice-detector
   ```
