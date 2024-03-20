import os
import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
data_folder = 'data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
left_eye_data = []
right_eye_data = []
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            if ex < w / 2: 
                left_eye_data.append(roi_gray[ey:ey + eh, ex:ex + ew])  # Left eye
            else:  
                right_eye_data.append(roi_gray[ey:ey + eh, ex:ex + ew]) # Right eye
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break #exit
cap.release()
cv2.destroyAllWindows()
for i, img in enumerate(left_eye_data):
    cv2.imwrite(os.path.join(data_folder, f'left_eye_{i}.png'), img)
for i, img in enumerate(right_eye_data):
    cv2.imwrite(os.path.join(data_folder, f'right_eye_{i}.png'), img)

print("Data collection completed.")
