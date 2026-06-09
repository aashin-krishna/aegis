@echo off
title Aegis Release Launcher & Setup Script
cd /d "%~dp0"

echo ====================================================================
echo    Aegis Security Intelligence Suite Launcher & Environment Setup
echo ====================================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not added to your system PATH.
    echo Please install Python 3.8+ from https://www.python.org/ and retry.
    echo.
    pause
    exit /b 1
)

:: 2. Check and setup virtual environment
if not exist ".venv" (
    echo [INFO] Virtual environment (.venv) not found. Creating one now...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create python virtual environment.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created.
    echo.
)

:: 3. Install requirements
echo [INFO] Checking / installing python dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
.venv\Scripts\python.exe -m pip install -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Failed to install python dependencies. Please check your internet connection and retry.
    pause
    exit /b 1
)
echo [SUCCESS] Python dependencies verified.
echo.

:: 4. Run Streamlit web app
echo [INFO] Starting Aegis Streamlit Web App on http://localhost:8550 ...
echo.
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8550

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Streamlit server stopped with exit code %ERRORLEVEL%.
    pause
)
