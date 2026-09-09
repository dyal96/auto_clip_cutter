@echo off
title AutoClipper Launcher
cd /d "%~dp0"

if not exist "Input" mkdir "Input"
if not exist "Output" mkdir "Output"

echo ===================================================
echo   🎬 Launching AutoClipper 9:16 Vertical Video Cutter
echo ===================================================
echo.

python app.py

pause
