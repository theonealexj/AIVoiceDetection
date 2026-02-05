@echo off
set "VENV_PYTHON=.\.venv\Scripts\python.exe"
set "VENV_PIP=.\.venv\Scripts\pip.exe"
set "UVICORN=.\.venv\Scripts\uvicorn.exe"

echo Using Virtual Environment at .\.venv

echo Installing dependencies...
"%VENV_PYTHON%" -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies
    exit /b %errorlevel%
)

echo Generating Data...
"%VENV_PYTHON%" training/generate_data.py
if %errorlevel% neq 0 (
    echo Failed to generate data
    exit /b %errorlevel%
)

echo Training Model...
"%VENV_PYTHON%" training/train.py
if %errorlevel% neq 0 (
    echo Failed to train model
    exit /b %errorlevel%
)

echo Testing API...
start "VoiceDetectorAPI" cmd /k ""%VENV_PYTHON%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
timeout /t 10
"%VENV_PYTHON%" test_api.py
