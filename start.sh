#!/bin/bash

# Activate the virtual environment created by Render's Python runtime
source /opt/render/project/src/.venv/bin/activate

# Run your application using uvicorn
# The PORT environment variable is automatically provided by Render
python3 -m uvicorn spice_summary_bot:application --host 0.0.0.0 --port $PORT
