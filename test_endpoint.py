"""
Test script for AI Voice Detection API Endpoint
This script helps you test your public endpoint before final evaluation
"""
import requests
import json
import base64

def test_endpoint(endpoint_url, api_key, audio_file_path):
    """
    Test the AI Voice Detection endpoint
    
    Args:
        endpoint_url: Full URL to your endpoint (e.g., https://abc123.ngrok.io/detect-voice)
        api_key: Your API key
        audio_file_path: Path to audio file to test with
    """
    print("=" * 60)
    print("AI Voice Detection API - Endpoint Test")
    print("=" * 60)
    
    # Read and encode audio file
    print(f"\n1. Reading audio file: {audio_file_path}")
    try:
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        print(f"   ✓ Audio encoded (length: {len(audio_base64)} chars)")
    except FileNotFoundError:
        print(f"   ✗ ERROR: File not found: {audio_file_path}")
        return
    
    # Prepare request
    print(f"\n2. Preparing request to: {endpoint_url}")
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Use exact field names from the API tester form
    payload = {
        "Language": "English",
        "Audio Format": "wav",
        "Audio Base64 Format": audio_base64
    }
    
    print(f"   Headers: x-api-key = {api_key}")
    print(f"   Body fields: Language, Audio Format, Audio Base64 Format")
    
    # Send request
    print(f"\n3. Sending POST request...")
    try:
        response = requests.post(
            endpoint_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"   Status Code: {response.status_code}")
        
        # Display response
        print(f"\n4. Response:")
        print("-" * 60)
        
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
            print("-" * 60)
            print("✓ SUCCESS! Your endpoint is working correctly.")
            print(f"\nClassification: {result.get('classification', 'N/A')}")
            print(f"Confidence: {result.get('confidenceScore', 'N/A')}")
        else:
            print(f"✗ ERROR: {response.status_code}")
            print(response.text)
            print("-" * 60)
            
    except requests.exceptions.ConnectionError:
        print("   ✗ ERROR: Could not connect to endpoint")
        print("   Make sure your server and ngrok are running")
    except requests.exceptions.Timeout:
        print("   ✗ ERROR: Request timed out")
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")

if __name__ == "__main__":
    # Configuration - Update these values
    ENDPOINT_URL = input("Enter your ngrok endpoint URL (e.g., https://abc123.ngrok.io/detect-voice): ").strip()
    API_KEY = "my-secret-api-key-2026"
    
    # Use sample audio file or specify your own
    AUDIO_FILE = "data/sample_audio.wav"  # Change this to your test file
    
    print(f"\nUsing API Key: {API_KEY}")
    print(f"Using Audio File: {AUDIO_FILE}")
    
    proceed = input("\nProceed with test? (y/n): ").strip().lower()
    if proceed == 'y':
        test_endpoint(ENDPOINT_URL, API_KEY, AUDIO_FILE)
    else:
        print("Test cancelled.")
