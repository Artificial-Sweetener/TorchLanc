@echo off
setlocal

REM Check if the virtual environment directory exists.
IF EXIST "venv\Scripts\activate.bat" (
    ECHO Found existing virtual environment.
) ELSE (
    ECHO No virtual environment found. Creating one...
    REM Create the virtual environment using the system's python.
    python -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        ECHO Failed to create virtual environment. Make sure Python is installed and in your PATH.
        GOTO :EOF
    )

    ECHO Activating virtual environment and installing dependencies...
    CALL venv\Scripts\activate.bat

    ECHO Installing TorchLanc in editable mode...
    pip install -e .
    IF %ERRORLEVEL% NEQ 0 (
        ECHO Failed to install project.
        GOTO :EOF
    )
    
    ECHO Installing benchmark-specific dependencies...
    pip install numpy torch torchvision Pillow py-cpuinfo
    IF %ERRORLEVEL% NEQ 0 (
        ECHO Failed to install benchmark dependencies.
        GOTO :EOF
    )
)

ECHO.
ECHO ==================================================================
ECHO Activating environment and running the benchmark race...
ECHO ==================================================================
ECHO.

REM Activate the venv and run the benchmark.
CALL venv\Scripts\activate.bat
python benchmark/benchmark.py --race

ECHO.
ECHO Benchmark complete. Press any key to exit.
pause > nul

endlocal
