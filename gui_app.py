import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from eye_detection import detect_eyes
from alert_system import trigger_alert, play_alert
from alert_system import trigger_alert
from cnn_eye_model import BasicCNN
from advanced_cnn_model import AdvancedCNN
from torchvision import transforms, models
import torch.nn as nn

model           = None
running         = False
closed_frames   = 0
closed_threshold= 15
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORMS = {
    "BasicCNN": transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((24, 24)),
        transforms.ToTensor()
    ]),
    "AdvancedCNN": transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((24, 24)),
        transforms.ToTensor()
    ]),
    "ResNet18": transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

def load_model(choice):
    global model, transform
    transform = TRANSFORMS[choice]
    if choice == "BasicCNN":
        model = BasicCNN()
        model.load_state_dict(torch.load("basic_cnn_eye.pth", map_location=DEVICE))
    elif choice == "AdvancedCNN":
        model = AdvancedCNN()
        model.load_state_dict(torch.load("advanced_cnn_eye.pth", map_location=DEVICE))
    elif choice == "ResNet18":
        r = models.resnet18(weights=None)
        in_f = r.fc.in_features
        r.fc = nn.Linear(in_f, 2)
        r.load_state_dict(torch.load("resnet18_best.pth", map_location=DEVICE))
        model = r
    else:
        raise ValueError(f"Unknown model choice: {choice}")

    model.to(DEVICE)
    model.eval()
    print(f"Loaded {choice} on {DEVICE}")

def predict_eye_state(eye_img):
    tensor = transform(eye_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        return int(torch.argmax(out, dim=1).item())

def start_detection():
    global running, closed_frames
    running = True
    closed_frames = 0
    cap = cv2.VideoCapture(0)

    def loop():
        global closed_frames
        if not running:
            cap.release()
            return

        ret, frame = cap.read()
        if not ret:
            return

        eyes = detect_eyes(frame)
        states = []
        for eye_side, eye_img, (ex, ey, ew, eh) in eyes:
            pred = predict_eye_state(eye_img)
            states.append(pred)
            label = "Open" if pred == 1 else "Closed"
            color = (0,255,0) if pred == 1 else (0,0,255)
            cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), color, 2)
            cv2.putText(frame, label, (ex, ey-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if states and states.count(0) == len(states):
            closed_frames += 1
        else:
            closed_frames = 0

        if closed_frames > closed_threshold:
            trigger_alert()
            status_label.config(text="*** WAKE UP! ***")
            closed_frames = 0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(10, loop)

    loop()

def stop_detection():
    global running
    running = False

root = tk.Tk()
root.title("Drowsiness Detection System")

model_var = tk.StringVar()
dropdown = ttk.Combobox(root, textvariable=model_var,
                        values=["BasicCNN","AdvancedCNN","ResNet18"],
                        state="readonly", width=20)
dropdown.current(0)
dropdown.pack(pady=10)
dropdown.bind("<<ComboboxSelected>>", lambda e: load_model(model_var.get()))

start_btn = tk.Button(root, text="Start Detection", command=start_detection)
start_btn.pack(pady=5)
stop_btn  = tk.Button(root, text="Stop Detection",  command=stop_detection)
stop_btn.pack(pady=5)

video_label = tk.Label(root)
video_label.pack()

status_label = tk.Label(root, text="", font=("Helvetica", 16), fg="red")
status_label.pack(pady=5)

load_model("BasicCNN")

root.mainloop()
