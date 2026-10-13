@echo off
REM Double-click to install/update and launch Bombovoice.
cd /d "%~dp0"
if not exist .venv (
    echo Setting up Bombovoice - first run takes a minute...
    py -3 -m venv .venv || python -m venv .venv
    .venv\Scripts\python -m pip install --upgrade pip
)
echo Checking for updates to Bombovoice's packages...
.venv\Scripts\python -m pip install -q -e ".[all]"
.venv\Scripts\python -m bombovoice %*
if errorlevel 1 pause
