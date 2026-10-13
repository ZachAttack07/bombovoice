@echo off
REM One-time setup for the GPU features: live AI voice (RVC) and voice cloning (XTTS).
REM Needs an NVIDIA GPU. RTX 50-series (e.g. RTX 5060) needs driver 570 or newer.
cd /d "%~dp0"
if not exist .venv (
    echo Run run_windows.bat once first to create the app environment.
    pause
    exit /b 1
)
echo.
echo === Installing PyTorch with CUDA (about 3 GB, takes a while) ===
REM CUDA 12.8+ builds are required for RTX 50-series (Blackwell) cards.
.venv\Scripts\python -m pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo CUDA 12.8 build not available, trying CUDA 13.0...
    .venv\Scripts\python -m pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu130
)
echo.
echo === Installing the AI voice + voice cloning packages ===
.venv\Scripts\python -m pip install -e ".[all,gpu]"
echo.
echo === Checking your GPU ===
.venv\Scripts\python -m bombovoice gpu-check
echo.
pause
