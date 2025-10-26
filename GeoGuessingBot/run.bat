@echo off
echo Starting program...
call "%~dp0.venv\Scripts\activate.bat"
"%~dp0.venv\Scripts\python.exe" -m streamlit run "app.py"
pause
