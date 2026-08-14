@echo off
echo Starting DCDD Annotation Tool...
echo.
echo The tool will open in your browser at http://localhost:8765/annotation_app/index.html
echo DO NOT close this window while annotating.
echo.
start http://localhost:8765/annotation_app/index.html
python -m http.server 8765
