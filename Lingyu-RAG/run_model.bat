@echo off
REM Start the Ollama server
start "" "D:\Coding\Ollama\ollama.exe" serve
timeout /t 5 /nobreak > NUL
REM Run the qwen3:8b model
"D:\Coding\Ollama\ollama.exe" run qwen3:8b