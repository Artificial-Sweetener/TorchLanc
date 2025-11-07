@echo off
setlocal EnableDelayedExpansion

set "VENV_DIR=venv"
set "ACTIVATE=%VENV_DIR%\Scripts\activate.bat"

if exist "%ACTIVATE%" (
    echo Found existing virtual environment.
    goto :AFTER_SETUP
)

echo No virtual environment found. Creating one...
python -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo Failed to create virtual environment. Make sure Python is installed and in PATH.
    goto :EOF
)

call "%ACTIVATE%"
call :INSTALL_REQUIREMENTS || goto :EOF
echo Setup complete.
goto :AFTER_SETUP

:AFTER_SETUP
echo.
echo ==================================================================
echo Activating environment and running the benchmark race...
echo ==================================================================
echo.
call "%ACTIVATE%"
python benchmark/benchmark.py --race || goto :EOF
echo.
echo Benchmark complete. Press any key to exit.
pause > nul
goto :EOF

:INSTALL_REQUIREMENTS
echo Installing dependencies inside the virtual environment...

set "TORCH_CUDA_INDEX_URL=%PYTORCH_CUDA_INDEX_URL%"
if "!TORCH_CUDA_INDEX_URL!"=="" set "TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/nightly/cu130"
set "TORCH_CPU_INDEX_URL=%PYTORCH_CPU_INDEX_URL%"
if "!TORCH_CPU_INDEX_URL!"=="" set "TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/nightly/cpu"

set "TORCH_INDEX_URL=!TORCH_CPU_INDEX_URL!"
set "TORCH_FLAVOR=CPU-only"
where nvidia-smi >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    nvidia-smi -L >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "TORCH_INDEX_URL=!TORCH_CUDA_INDEX_URL!"
        set "TORCH_FLAVOR=CUDA-enabled"
    )
)

echo Installing PyTorch !TORCH_FLAVOR! from !TORCH_INDEX_URL!
pip install numpy Pillow py-cpuinfo || exit /b 1
pip install --upgrade torch torchvision --index-url !TORCH_INDEX_URL! || exit /b 1

call :VERIFY_CUDA
if errorlevel 1 (
    echo CUDA unusable; reinstalling CPU-only PyTorch.
    pip uninstall -y torch torchvision || exit /b 1
    pip install --upgrade torch torchvision --index-url !TORCH_CPU_INDEX_URL! || exit /b 1
)

echo Installing TorchLanc in editable mode...
pip install -e . || exit /b 1
exit /b 0

:VERIFY_CUDA
set "_CUDA_CHECK=%TEMP%\torch_cuda_check.py"
>"%_CUDA_CHECK%" echo import sys
>>"%_CUDA_CHECK%" echo import torch
>>"%_CUDA_CHECK%" echo if not torch.cuda.is_available():
>>"%_CUDA_CHECK%" echo ^    sys.exit^(1^)
>>"%_CUDA_CHECK%" echo try:
>>"%_CUDA_CHECK%" echo ^    torch.zeros^(1, device='cuda'^).sum^(^).item^(^)
>>"%_CUDA_CHECK%" echo except Exception:
>>"%_CUDA_CHECK%" echo ^    sys.exit^(2^)
>>"%_CUDA_CHECK%" echo sys.exit^(0^)
"%VENV_DIR%\Scripts\python" "%_CUDA_CHECK%"
set "CUDA_CHECK_ERROR=%ERRORLEVEL%"
del "%_CUDA_CHECK%" >nul 2>&1
exit /b %CUDA_CHECK_ERROR%
