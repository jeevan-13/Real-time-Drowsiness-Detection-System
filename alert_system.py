import os
import time
import threading
import platform

# Alert with sound: simply uses system voice or beep

def play_alert():
    system = platform.system()
    if system == 'Windows':
        import winsound
        winsound.Beep(1000, 500)  # frequency, duration
    elif system == 'Darwin':  # macOS
        os.system('say "Wake up!"')
    else:  # Linux and others
        os.system('say "Wake up!"')  # fallback using 'say' if available

# Log alert with timestamp
def log_alert(logfile="alert_log.txt"):
    with open(logfile, "a") as f:
        f.write(f"[ALERT] Drowsiness detected at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Trigger both in a separate thread to not block GUI
def trigger_alert():
    threading.Thread(target=play_alert, daemon=True).start()
    log_alert()
