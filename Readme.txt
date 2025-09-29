==========================================================
REAL-TIME DROWSINESS DETECTION SYSTEM
==========================================================

Project Description:
---------------------
This project implements a real-time driver drowsiness detection system using deep learning and computer vision techniques. The system continuously monitors eye states (open or closed) from live webcam footage and triggers an alert when it detects signs of drowsiness.

Three models were used for classification:
1. Basic CNN – Lightweight, fast, and the most accurate (97%)
2. Advanced CNN – Deeper architecture with 95% accuracy
3. ResNet18 – Transfer learning model, achieved 83% accuracy

Key Features:
-------------
- Real-time eye detection using Haar Cascades
- Three selectable models via GUI
- Sound alert when drowsiness is detected
- Logs all drowsiness events with timestamps
- Runs efficiently on standard laptops (no GPU required)

File Descriptions:
------------------
Code/ 
- gui_app.py ............ Main application with GUI
- cnn_eye_model.py ...... Basic CNN model
- advanced_cnn_model.py . Advanced CNN model
- resnet_model.py ....... Transfer learning model using ResNet18
- eye_detection.py ...... Eye detection using OpenCV
- alert_system.py ....... Sound alert and log file system
- run.sh ................ Shell script to launch the application
- requirements.txt ...... Python dependencies
- train.................. Dataset

How to Run:
-----------
1. Ensure Python 3.8+ is installed.
2. (Optional but recommended) Create a virtual environment:
       python3 -m venv venv
       source venv/bin/activate
3. Install dependencies:
       pip install -r requirements.txt
4. Run the application:
       python gui_app.py
   OR using the shell script (if Python path is configured correctly):
       ./run.sh

Dependencies:
-------------
Install the following packages if not using requirements.txt:
- opencv-python
- numpy
- pillow
- torch
- torchvision
- tkinter (comes with Python)
- pygame (for alert sound)

Output:
-------
- A GUI window with webcam feed and live eye detection
- Dropdown to select model (Basic CNN, Advanced CNN, ResNet18)
- Real-time alert sound if drowsiness detected
- Log saved in `alert_log.txt` with event timestamps

Authors:
--------
- Jeevan Kumar Reddy Palicherla
- Goutham Balla