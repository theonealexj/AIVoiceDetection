@echo off
echo =========================================
echo AI Voice Detection - Public Endpoint Setup
echo =========================================
echo.

REM Check if ngrok is installed
where ngrok >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: ngrok is not installed!
    echo.
    echo Please install ngrok:
    echo 1. Go to https://ngrok.com/download
    echo 2. Download ngrok for Windows
    echo 3. Extract ngrok.exe to a location in your PATH
    echo    OR place it in this project directory
    echo.
    pause
    exit /b 1
)

echo =========================================
echo Starting FastAPI Server...
echo =========================================
start "AI Voice API Server" cmd /k ".\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

echo Waiting for server to start...
timeout /t 5 /nobreak >nul

echo.
echo =========================================
echo Starting ngrok tunnel...
echo =========================================
echo.
echo Your API will be available at the ngrok URL shown below.
echo Copy the HTTPS URL and use it in the API tester.
echo.
echo Your API Key (for x-api-key header): my-secret-api-key-2026
echo.
echo =========================================
ngrok http 8000
