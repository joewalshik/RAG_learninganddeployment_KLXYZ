@echo off
REM Navigate to your project directory
cd /d "D:\Coding\Ollama\rag_app"
REM Activate the virtual environment
call "D:\Coding\Ollama\rag_app\rag_venv\Scripts\activate.bat"
REM Run the Streamlit app
streamlit run app.py
pause
