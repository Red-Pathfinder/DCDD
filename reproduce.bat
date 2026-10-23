@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "PYTHON=python"

if exist "%ROOT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT_DIR%.venv\Scripts\python.exe"
) else if exist "%ROOT_DIR%Drift\.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT_DIR%Drift\.venv\Scripts\python.exe"
)

"%PYTHON%" "%ROOT_DIR%reproduce.py" %*
exit /b %errorlevel%
