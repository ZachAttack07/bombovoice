@echo off
REM Double-click to install (first time) and launch Bombovoice.
cd /d "%~dp0"
if not exist .venv (
    echo Setting up Bombovoice - first run takes a minute...
    py -3 -m venv .venv || python -m venv .venv
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -e ".[all]"
)
.venv\Scripts\python -m bombovoice %*
if errorlevel 1 pause
