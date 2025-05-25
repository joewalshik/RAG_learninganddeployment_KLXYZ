@echo off
cd /d "D:\Coding\Ollama\rag_app"
call "D:\Coding\Ollama\rag_app\rag_venv\Scripts\activate.bat"
python process_documents.py
pause