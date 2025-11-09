import cv2
import mediapipe as mp
import pyautogui
import os
import numpy as np
from collections import deque
from pynput import keyboard
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# Load environment variables
load_dotenv()

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# Initialize Roboflow client
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not found in environment variables. Please set it in .env file")

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)
MODEL_ID = "tongue-wltgn/2"

# Webcam
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

# 4 card positions
card_1_x = 622
card_1_y = 752

card_2_x = 684
card_2_y = 752

card_3_x = 752
card_3_y = 752

card_4_x = 819
card_4_y = 752

# Middle center position
middle_center_x = 691
middle_center_y = 481

# Mouse control settings
SENSITIVITY = 0.5  # mouse movement sensitivity
SMOOTHING_FRAMES = 5  # Number of frames to average for smoother movement
MIN_CONFIDENCE = 0.5  # Minimum confidence threshold for tongue detection
FRAMES_PER_INFERENCE = 1  # Number of frames to collect before running inference (for accuracy)

# Vector smoothing buffer
vector_buffer = []
# Frame buffer for multi-frame inference
frame_buffer = []
last_tongue_tip = None  # Keep last detection while collecting new frames
mouth_was_open = False
current_card = 1  # Track current card (1-4)
current_mouse_x, current_mouse_y = card_1_x, card_1_y
pyautogui.moveTo(current_mouse_x, current_mouse_y)

# Global keyboard input handling
key_press_queue = deque(maxlen=10)  # Thread-safe queue for key presses
keyboard_listener = None
running = True

def on_key_press(key):
    """Handle global key press events"""
    try:
        if hasattr(key, 'char') and key.char:
            key_press_queue.append(key.char.lower())
    except AttributeError:
        pass

def start_keyboard_listener():
    """Start the global keyboard listener in a separate thread"""
    global keyboard_listener
    keyboard_listener = keyboard.Listener(on_press=on_key_press)
    keyboard_listener.start()
    return keyboard_listener

def get_card_position(card_num):
    """Get the x, y coordinates for a given card number (1-4)"""
    if card_num == 1:
        return card_1_x, card_1_y
    elif card_num == 2:
        return card_2_x, card_2_y
    elif card_num == 3:
        return card_3_x, card_3_y
    elif card_num == 4:
        return card_4_x, card_4_y
    else:
        return card_1_x, card_1_y  # Default to card 1

def detect_tongue_tip(frame):
    """Run Roboflow inference to detect tongue tip coordinates on a single frame"""
    try:
        result = CLIENT.infer(frame, model_id=MODEL_ID)
        if result and "predictions" in result and len(result["predictions"]) > 0:
            prediction = result["predictions"][0]
            confidence = prediction.get("confidence", 0)
            if confidence >= MIN_CONFIDENCE:
                # Roboflow returns coordinates in image pixel space
                tongue_x = prediction.get("x", 0)
                tongue_y = prediction.get("y", 0)
                return tongue_x, tongue_y, confidence
    except Exception as e:
        print(f"Inference error: {e}")
    return None, None, 0

def detect_tongue_tip_multi_frame(frames):
    """Run Roboflow inference on multiple frames and average the results for better accuracy"""
    if not frames:
        return None, None, 0
    
    detections = []
    confidences = []
    
    # Run inference on each frame
    for frame in frames:
        tongue_x, tongue_y, confidence = detect_tongue_tip(frame)
        if tongue_x is not None and tongue_y is not None:
            detections.append((tongue_x, tongue_y))
            confidences.append(confidence)
    
    # Average the detections
    if len(detections) > 0:
        avg_x = np.mean([d[0] for d in detections])
        avg_y = np.mean([d[1] for d in detections])
        avg_confidence = np.mean(confidences)
        return avg_x, avg_y, avg_confidence
    
    return None, None, 0

def calculate_vector(mouth_center, tongue_tip, frame_shape):
    """Calculate vector from mouth center to tongue tip"""
    if tongue_tip[0] is None or tongue_tip[1] is None:
        return None
    
    h, w = frame_shape[:2]
    mouth_x, mouth_y = mouth_center
    tongue_x, tongue_y = tongue_tip
    
    # Calculate vector (tongue - mouth)
    vector_x = tongue_x - mouth_x
    vector_y = tongue_y - mouth_y
    
    return (vector_x, vector_y)

def smooth_vector(new_vector):
    """Add vector to buffer and return averaged vector"""
    if new_vector is None:
        return None
    
    vector_buffer.append(new_vector)
    
    # Keep only last SMOOTHING_FRAMES vectors
    if len(vector_buffer) > SMOOTHING_FRAMES:
        vector_buffer.pop(0)
    
    # Calculate average vector
    if len(vector_buffer) > 0:
        avg_x = np.mean([v[0] for v in vector_buffer])
        avg_y = np.mean([v[1] for v in vector_buffer])
        return (avg_x, avg_y)
    
    return None

def move_mouse_from_vector(vector, frame_shape):
    """Translate vector to mouse movement"""
    if vector is None:
        return
    
    global current_mouse_x, current_mouse_y
    
    h, w = frame_shape[:2]
    
    # Normalize vector to screen coordinates
    # Scale based on frame size to screen size
    scale_x = (screen_w / w) * SENSITIVITY
    scale_y = (screen_h / h) * SENSITIVITY
    
    # Calculate mouse movement delta
    delta_x = vector[0] * scale_x
    delta_y = vector[1] * scale_y
    
    # Update mouse position
    new_x = int(current_mouse_x + delta_x)
    new_y = int(current_mouse_y + delta_y)
    
    # Clamp to screen boundaries
    new_x = max(0, min(screen_w - 1, new_x))
    new_y = max(0, min(screen_h - 1, new_y))
    
    # Move mouse
    pyautogui.moveTo(new_x, new_y, duration=0.01)
    current_mouse_x, current_mouse_y = new_x, new_y

# Start global keyboard listener
print("Starting global keyboard listener...")
start_keyboard_listener()
print("Keyboard listener started. Press 'A' and 'D' keys globally to navigate cards.")

# Main loop
while running:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    h, w, _ = frame.shape
    tongue_tip_x, tongue_tip_y, tongue_confidence = None, None, 0
    mouth_center_px = None
    vector = None
    mouth_is_closed = False  # Track if mouth is closed in current frame

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
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
            cv2.circle(frame, mouth_center_px, 8, (0, 255, 0), -1)
            cv2.circle(frame, mouth_center_px, 10, (0, 255, 0), 2)

            upper_y = int(upper_lip.y * h)
            lower_y = int(lower_lip.y * h)
            mouth_open_distance = lower_y - upper_y
            mouth_is_open = mouth_open_distance >= 8

            # If mouth is open, detect tongue and control mouse
            if mouth_is_open:
                mouth_is_closed = False
                # Clear any accumulated key presses when mouth opens
                key_press_queue.clear()
                mouth_was_open = True
                
                # Add frame to buffer for multi-frame inference
                frame_buffer.append(frame.copy())
                
                # Keep only last FRAMES_PER_INFERENCE frames
                if len(frame_buffer) > FRAMES_PER_INFERENCE:
                    frame_buffer.pop(0)
                
                # Run inference when we have enough frames
                if len(frame_buffer) >= FRAMES_PER_INFERENCE:
                    tongue_tip_x, tongue_tip_y, tongue_confidence = detect_tongue_tip_multi_frame(frame_buffer)
                    if tongue_tip_x is not None and tongue_tip_y is not None:
                        last_tongue_tip = (tongue_tip_x, tongue_tip_y, tongue_confidence)
                    # Clear buffer after inference to get fresh frames
                    frame_buffer.clear()
                elif last_tongue_tip is not None:
                    # Use last detection while collecting new frames
                    tongue_tip_x, tongue_tip_y, tongue_confidence = last_tongue_tip
                
                # If we have tongue detection, calculate vector and move mouse
                if tongue_tip_x is not None and tongue_tip_y is not None and mouth_center_px:
                    # Draw tongue tip
                    tongue_tip_px = (int(tongue_tip_x), int(tongue_tip_y))
                    cv2.circle(frame, tongue_tip_px, 8, (255, 0, 0), -1)  # Red dot for tongue
                    cv2.circle(frame, tongue_tip_px, 10, (255, 0, 0), 2)
                    
                    # Draw vector line
                    cv2.line(frame, mouth_center_px, tongue_tip_px, (0, 255, 255), 2)
                    
                    # Calculate vector
                    vector = calculate_vector(mouth_center_px, (tongue_tip_x, tongue_tip_y), frame.shape)
                    
                    # Smooth vector using multiple frames
                    smoothed_vector = smooth_vector(vector)
                    
                    # Move mouse based on smoothed vector
                    if smoothed_vector:
                        move_mouse_from_vector(smoothed_vector, frame.shape)
                    
                    # Display info
                    cv2.putText(frame, f"Tongue detected (conf: {tongue_confidence:.2f})", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if vector:
                        cv2.putText(frame, f"Vector: ({vector[0]:.1f}, {vector[1]:.1f})", (30, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "MOUTH OPEN - Detecting tongue...", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                # Mouth closed - move mouse to card position
                mouth_is_closed = True
                if mouth_was_open:
                    mouth_was_open = False
                    vector_buffer.clear()  # Clear smoothing buffer
                    frame_buffer.clear()  # Clear frame buffer
                    last_tongue_tip = None  # Clear last detection
                    current_card = 1  # Reset to card 1 when mouth closes
                    current_mouse_x, current_mouse_y = get_card_position(current_card)
                    pyautogui.moveTo(current_mouse_x, current_mouse_y)
                
                # Display current card info
                cv2.putText(frame, f"MOUTH CLOSED - CARD {current_card}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "Press 'A' for left, 'D' for right", (30, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No face detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Tongue Mouse Control", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Handle keyboard input for card navigation (only when mouth is closed)
    # Check global keyboard input queue
    if mouth_is_closed and len(key_press_queue) > 0:
        pressed_key = key_press_queue.popleft()  # Get and remove the first key from queue
        if pressed_key == 'a':
            # Move to previous card (left)
            if current_card > 1:
                current_card -= 1
                current_mouse_x, current_mouse_y = get_card_position(current_card)
                pyautogui.moveTo(current_mouse_x, current_mouse_y)
                print(f"Moved to Card {current_card}")
        elif pressed_key == 'd':
            # Move to next card (right)
            if current_card < 4:
                current_card += 1
                current_mouse_x, current_mouse_y = get_card_position(current_card)
                pyautogui.moveTo(current_mouse_x, current_mouse_y)
                print(f"Moved to Card {current_card}")
    
    if key == 27:  # ESC key (still works from OpenCV window)
        running = False
        break

# Cleanup
running = False
if keyboard_listener:
    keyboard_listener.stop()
cap.release()
cv2.destroyAllWindows()