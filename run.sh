#!/bin/bash

echo "------------------------------------------------------"
echo "Starting the Drowsiness Detection System Application..."
echo "------------------------------------------------------"

# Step 1: Check and install required Python packages
if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "[WARNING] requirements.txt not found. Skipping package installation."
fi

# Step 2: Run the main GUI application
if [ -f "gui_app.py" ]; then
    echo "[INFO] Launching the GUI..."
    python gui_app.py
else
    echo "[ERROR] gui_app.py not found. Please check your files."
fi

echo "[INFO] Application closed. Goodbye!"
