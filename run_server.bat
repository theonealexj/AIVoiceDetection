@echo off
set "VENV_PYTHON=.\.venv\Scripts\python.exe"

echo Starting Server using .venv...
"%VENV_PYTHON%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
