@echo off
title Stock Analyser
echo Starting Stock Analyser...
echo.

:: Add Python to PATH
set "PATH=%LOCALAPPDATA%\Python\pythoncore-3.14-64\Scripts;%LOCALAPPDATA%\Python\pythoncore-3.14-64;%LOCALAPPDATA%\Microsoft\WindowsApps;%PATH%"

:: Change to app directory
cd /d "%~dp0"

:: Start Streamlit
py -m streamlit run app.py --server.headless false

pause
