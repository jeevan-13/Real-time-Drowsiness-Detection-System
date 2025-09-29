import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    eyes = []

    if len(faces) == 0:
        return eyes

    x, y, w, h = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]

    left_eyes = left_eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
    right_eyes = right_eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

    if len(left_eyes) > 0:
        ex, ey, ew, eh = max(left_eyes, key=lambda b: b[2] * b[3])
        eye_img = roi_color[ey:ey+eh, ex:ex+ew]
        eyes.append(('left', eye_img, (x+ex, y+ey, ew, eh)))

    if len(right_eyes) > 0:
        ex, ey, ew, eh = max(right_eyes, key=lambda b: b[2] * b[3])
        eye_img = roi_color[ey:ey+eh, ex:ex+ew]
        eyes.append(('right', eye_img, (x+ex, y+ey, ew, eh)))

    return eyes
