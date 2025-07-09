import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Load known face images
path = 'known_faces'
images = []
names = []

print("\nLoading known face images...")
for filename in os.listdir(path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(f'{path}/{filename}')
        images.append(img)
        names.append(os.path.splitext(filename)[0])  # use filename without extension
        print(f"Loaded: {filename}")

# Function to encode known faces
def find_encodings(images):
    encode_list = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if encodings:
            encode_list.append(encodings[0])
    return encode_list

known_encodings = find_encodings(images)
print("\nEncoding complete.")

# Function to mark attendance
def mark_attendance(name):
    with open('attendance.csv', 'a+') as f:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'{name},{now}\n')
        print(f"Marked attendance for: {name} at {now}")

# Start webcam
print("\nStarting webcam...")
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    img_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Resize for faster processing
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    # Detect faces and encodings in the current frame
    faces_cur_frame = face_recognition.face_locations(img_rgb)
    encodes_cur_frame = face_recognition.face_encodings(img_rgb, faces_cur_frame)

    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
        matches = face_recognition.compare_faces(known_encodings, encode_face)
        face_dist = face_recognition.face_distance(known_encodings, encode_face)

        match_index = np.argmin(face_dist)
        match_index = np.argmin(face_dist)
        if matches[match_index]:
            name = names[match_index].upper()
            print(f"Detected: {name}")
            mark_attendance(name)

            # Draw a rectangle and label on the face
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the camera frame
    cv2.imshow('Webcam', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
