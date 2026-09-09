@echo off
title AutoClipper Studio - Cloud Web Server
echo ========================================================
echo Starting AutoClipper Studio Web UI...
echo ========================================================
python web_app.py --server-name 0.0.0.0 --port 7860 --share --inbrowser
pause
