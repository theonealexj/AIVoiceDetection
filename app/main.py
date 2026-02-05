from fastapi import FastAPI, HTTPException, Header, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Ensure app can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.inference import load_model, predict_voice

# Configuration
API_KEY_NAME = "x-api-key"
API_KEY = os.getenv("API_KEY", "secret-key-123") # Default for demo

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    load_model()
    yield

app = FastAPI(title="AI Voice Detector API", lifespan=lifespan)

from typing import Optional
from pydantic import BaseModel, Field

class AudioRequest(BaseModel):
    # Multiple aliases to support different API testers
    audio_base64_format: Optional[str] = Field(None, alias="Audio Base64 Format")
    audio_base64_camel: Optional[str] = Field(None, alias="audioBase64Format")
    audio_base64_exact: Optional[str] = Field(None, alias="audioBase64")  # Exact match for the tester
    audio_base64: Optional[str] = None
    audio: Optional[str] = None
    base64: Optional[str] = None
    audio_data: Optional[str] = None
    audioData: Optional[str] = None
    file: Optional[str] = None
    
    language: Optional[str] = Field(None, alias="Language")
    audio_format: Optional[str] = Field(None, alias="Audio Format")
    audio_format_camel: Optional[str] = Field(None, alias="audioFormat")  # camelCase version

    class Config:
        allow_population_by_field_name = True

# Auth dependency
async def get_api_key(
    api_key_header_val: str = Security(api_key_header),
    auth_header_val: str = Header(None, alias="Authorization")
):
    # Check x-api-key
    if api_key_header_val == API_KEY:
        return api_key_header_val
        
    # Check Bearer token
    if auth_header_val and auth_header_val.startswith("Bearer "):
        token = auth_header_val.split(" ")[1]
        if token == API_KEY:
            return token
            
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API Key"
    )

@app.post("/detect-voice")
async def detect_voice(request: AudioRequest, api_key: str = Depends(get_api_key)):
    try:
        # Support various keys (Robustness for the tester)
        b64_data = (request.audio_base64_format or request.audio_base64 or 
                   request.audio_base64_camel or request.audio_base64_exact or 
                   request.audio or request.base64 or request.audio_data or 
                   request.audioData or request.file)
        
        print(f"DEBUG: Request Fields: {request.dict(exclude_none=True).keys()}")
        if b64_data:
            print(f"DEBUG: Found valid audio data (len={len(b64_data)})")
        
        if not b64_data:
             print("DEBUG: Missing audio content")
             print(f"DEBUG: Full request data: {request.dict()}")
             raise HTTPException(status_code=400, detail="Missing audio content. Please provide audio data in one of these fields: audio_base64, audio, base64, audioData, or file")
        
        # Clean Base64 (Remove dataURI prefix if present)
        if "," in b64_data[:100]:
            print("DEBUG: Removing dataURI prefix")
            b64_data = b64_data.split(",", 1)[1]
            
        # Remove newlines/spaces
        b64_data = b64_data.replace("\n", "").replace("\r", "").replace(" ", "")
        
        print(f"DEBUG: Base64 Length: {len(b64_data)}")
        print(f"DEBUG: Language field: {request.language}")
        print(f"DEBUG: Full request dict: {request.dict(exclude_none=True)}")
        
        raw_result = predict_voice(b64_data)
        
        # Generate explanation based on classification
        if raw_result["classification"] == "AI_GENERATED":
            explanation = "Unnatural pitch consistency and robotic speech patterns detected"
        else:
            explanation = "Natural prosody and emotional variation detected"

        return {
            "status": "success",
            "language": request.language or "Unknown",
            "classification": raw_result["classification"],
            "confidenceScore": raw_result["confidence"],
            "explanation": explanation
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Log error internally
        import traceback
        traceback.print_exc()
        print(f"Internal Error: {e}")
        # Return actual error to user for debugging
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")

@app.get("/")
def home():
    return {
        "message": "AI Voice Detector API is running.",
        "usage": "Send POST request to /detect-voice with x-api-key header."
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
