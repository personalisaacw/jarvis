@echo off
cd /d "C:\OllamaThinkRouter"
call venv\Scripts\activate.bat
uvicorn router:app --host 127.0.0.1 --port 11435
