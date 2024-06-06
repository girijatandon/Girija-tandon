# Augmented-Reality-Mobile-Device-Security
ARSecure
import cv2
import numpy as np

# Load pre-trained Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Function to detect motion using frame differencing
def detect_motion(prev_gray, curr_gray):
    frame_diff = cv2.absdiff(prev_gray, curr_gray)
    _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    motion_pixels = np.count_nonzero(motion_mask)
    return motion_pixels > 1000

# Function to detect if eyes are open
def detect_eyes_open(eyes):
    return len(eyes) > 0

# Function to detect smiles
def detect_smile(smiles):
    return len(smiles) > 0

# Function to check for head movement by comparing face bounding boxes
def detect_head_movement(prev_faces, curr_faces):
    if len(prev_faces) == 0 or len(curr_faces) == 0:
        return False
    prev_face = prev_faces[0]
    curr_face = curr_faces[0]
    prev_center = (prev_face[0] + prev_face[2] // 2, prev_face[1] + prev_face[3] // 2)
    curr_center = (curr_face[0] + curr_face[2] // 2, curr_face[1] + curr_face[3] // 2)
    distance = np.sqrt((prev_center[0] - curr_center[0]) ** 2 + (prev_center[1] - curr_center[1]) ** 2)
    return distance > 10

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize previous frame and faces
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_faces = []

# Main loop for liveness detection
eye_counter = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Flags to track if any motion, eye blinking, smiling, or head movement is detected
    motion_detected = False
    eyes_detected = False
    smile_detected = False
    head_movement_detected = False

    # Process each detected face
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eye blinking
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if detect_eyes_open(eyes):
            eyes_detected = True
            eye_counter += 1

        # Reset eye counter if no eyes detected
        if eye_counter > 10:
            eyes_detected = False
            eye_counter = 0

        # Detect smiling
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
        if detect_smile(smiles):
            smile_detected = True

        # Check for head movement
        if detect_head_movement(prev_faces, faces):
            head_movement_detected = True

        # Check for motion
        if detect_motion(prev_gray, gray):
            motion_detected = True

    # Update the previous frame and faces
    prev_gray = gray.copy()
    prev_faces = faces

    # Determine liveness
    if motion_detected or eyes_detected or smile_detected or head_movement_detected:
        cv2.putText(frame, "Real", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Fake", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Liveness Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

