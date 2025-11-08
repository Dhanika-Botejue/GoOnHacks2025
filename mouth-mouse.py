import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Get mouth landmarks
            upper_lip = face_landmarks.landmark[13]  # Upper lip center
            lower_lip = face_landmarks.landmark[14]  # Lower lip center
            right_corner = face_landmarks.landmark[61]  # Right mouth corner
            left_corner = face_landmarks.landmark[291]  # Left mouth corner

            # Calculate mouth center (average of key mouth points)
            mouth_center_x = (upper_lip.x + lower_lip.x + right_corner.x + left_corner.x) / 4
            mouth_center_y = (upper_lip.y + lower_lip.y + right_corner.y + left_corner.y) / 4

            # Convert to pixel coordinates
            mouth_center_px = (int(mouth_center_x * w), int(mouth_center_y * h))

            # Draw green dot at mouth center
            cv2.circle(frame, mouth_center_px, 8, (0, 255, 0), -1)  # Green filled circle
            cv2.circle(frame, mouth_center_px, 10, (0, 255, 0), 2)  # Green outline circle

            upper_y = int(upper_lip.y * h)
            lower_y = int(lower_lip.y * h)

            mouth_open_distance = lower_y - upper_y

            # If mouth is nearly closed (small gap)
            if mouth_open_distance < 8:
                # Move mouse to center of screen
                pyautogui.moveTo(screen_w // 2, screen_h // 2)
                cv2.putText(frame, "MOUTH CLOSED - MOUSE CENTERED", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "MOUTH OPEN", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Mouth Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()