#!/bin/bash
echo "Starting DCDD Annotation Tool..."
echo ""
echo "The tool will open in your browser at http://localhost:8765/annotation_app/index.html"
echo "DO NOT close this terminal window while annotating."
echo ""
cd "$(dirname "$0")"
open "http://localhost:8765/annotation_app/index.html" 2>/dev/null || xdg-open "http://localhost:8765/annotation_app/index.html" 2>/dev/null
python3 -m http.server 8765
