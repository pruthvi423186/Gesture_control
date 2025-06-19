import mediapipe as mp
import cv2
import numpy as np
import time
import pyautogui
import subprocess
import platform
import webbrowser

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Imports for drawing landmarks
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Global variables to hold the latest results and the annotated image
latest_result = None
annotated_image_global = None

# Global variables for mouse control
prev_mouse_x = None
prev_mouse_y = None
mouse_control_active = False

# Global variable to track the last triggered gesture to prevent repeated actions
last_triggered_gesture = "None"
last_gesture_timestamp = 0

# Confidence threshold (90%)
CONFIDENCE_THRESHOLD = 0.70

# Mouse sensitivity
MOUSE_SENSITIVITY = 2.0

def print_and_draw_results(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    """
    Callback function to process gesture recognition results and trigger system actions.
    """
    global latest_result, annotated_image_global, prev_mouse_x, prev_mouse_y, mouse_control_active
    global last_triggered_gesture, last_gesture_timestamp
    
    latest_result = result
    
    # List of allowed gestures from the first code
    allowed_gestures = [
        "none", 
        "A", "a", "B", "b", "C", "c", "D", "d", "E", "e", "F", "f", "G", "g",
        "H", "h", "I", "i", "J", "j", "K", "k", "L", "l", "M", "m", "N", "n",
        "O", "o", "P", "p", "Q", "q", "R", "r", "S", "s", "T", "t", "U", "u",
        "V", "v", "W", "w", "X", "x", "Y", "y", "Z", "z"
    ]
    
    # Start with a copy of the input image for annotation
    annotated_image = np.copy(output_image.numpy_view())

    # Determine the current top-recognized gesture with confidence check
    current_gesture = "none"
    current_confidence = 0.0
    
    if result.gestures and result.gestures[0]:
        gesture_category = result.gestures[0][0].category_name
        gesture_confidence = result.gestures[0][0].score
        
        # Only accept gestures with 90% or higher confidence
        if gesture_confidence >= CONFIDENCE_THRESHOLD and gesture_category in allowed_gestures:
            current_gesture = gesture_category
            current_confidence = gesture_confidence

    # --- System Control Logic based on Gesture ---
    current_time = time.time()
    
    # Check if enough time has passed and gesture has changed to allow new action
    if (current_gesture != last_triggered_gesture and 
        current_gesture != "none" and 
        current_confidence >= CONFIDENCE_THRESHOLD and
        (current_time - last_gesture_timestamp) > 1.0):  # 1 second cooldown
        
        if current_gesture.upper() == "C" or current_gesture.lower() == "c":
            print(f"Action: Open Edge (Confidence: {current_confidence:.2f})")
            try:
                if platform.system() == "Windows":
                    subprocess.Popen(['start', 'msedge'], shell=True)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.Popen(["open", "-a", "Microsoft Edge"])
                elif platform.system() == "Linux":
                    subprocess.Popen(["microsoft-edge"])
            except Exception as e:
                print(f"Error opening Edge: {e}")
            last_triggered_gesture = current_gesture
            last_gesture_timestamp = current_time
            
        elif current_gesture.upper() == "L" or current_gesture.lower() == "l":
            print(f"Action: Maximize/Restore Current Window (Confidence: {current_confidence:.2f})")
            try:
                # Use Alt+Space to open window menu, then 'x' for maximize or 'r' for restore
                # Or use F11 for fullscreen toggle, or Windows+Up for maximize
                pyautogui.hotkey('win', 'up')  # This will maximize if not maximized, restore if maximized
            except Exception as e:
                print(f"Error toggling window size: {e}")
            last_triggered_gesture = current_gesture
            last_gesture_timestamp = current_time
            
        elif current_gesture.upper() == "O" or current_gesture.lower() == "o":
            print(f"Action: Open File Explorer (Confidence: {current_confidence:.2f})")
            try:
                if platform.system() == "Windows":
                    subprocess.Popen(["explorer.exe"])
                elif platform.system() == "Darwin":  # macOS
                    subprocess.Popen(["open", "."])
                elif platform.system() == "Linux":
                    subprocess.Popen(["xdg-open", "."])
            except Exception as e:
                print(f"Error opening file explorer: {e}")
            last_triggered_gesture = current_gesture
            last_gesture_timestamp = current_time
            
        elif current_gesture.upper() == "W" or current_gesture.lower() == "w":
            print(f"Action: Open YouTube in Chrome Guest Mode (Confidence: {current_confidence:.2f})")
            try:
                if platform.system() == "Windows":
                    # Try different Chrome executable paths
                    chrome_paths = [
                        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                        "chrome"
                    ]
                    
                    chrome_opened = False
                    for chrome_path in chrome_paths:
                        try:
                            subprocess.Popen([
                                chrome_path,
                                "--guest",
                                "--new-window",
                                "https://www.youtube.com"
                            ])
                            chrome_opened = True
                            break
                        except FileNotFoundError:
                            continue
                    
                    if not chrome_opened:
                        # Fallback: use webbrowser module
                        webbrowser.open("https://www.youtube.com")
                        
                elif platform.system() == "Darwin":  # macOS
                    subprocess.Popen([
                        "open", "-a", "Google Chrome",
                        "--args", "--guest", "--new-window", "https://www.youtube.com"
                    ])
                elif platform.system() == "Linux":
                    subprocess.Popen([
                        "google-chrome",
                        "--guest",
                        "--new-window",
                        "https://www.youtube.com"
                    ])
            except Exception as e:
                print(f"Error opening YouTube in Chrome guest mode: {e}")
                # Fallback to default browser
                try:
                    webbrowser.open("https://www.youtube.com")
                    print("Opened YouTube in default browser as fallback")
                except:
                    print("Failed to open YouTube")
            last_triggered_gesture = current_gesture
            last_gesture_timestamp = current_time
            
        elif current_gesture.upper() == "V" or current_gesture.lower() == "v":
            print(f"Action: Open Command Prompt and Activate Virtual Environment (Confidence: {current_confidence:.2f})")
            try:
                if platform.system() == "Windows":
                    # Method 1: Direct command execution
                    cmd_command = 'start cmd /k "cd /d D: && D:\\mp-310\\Scripts\\activate.bat"'
                    subprocess.Popen(cmd_command, shell=True)
                    
                else:
                    print("Virtual environment activation is configured for Windows only")
            except Exception as e:
                print(f"Error opening command prompt and activating virtual environment: {e}")
                # Alternative method if the first fails
                try:
                    if platform.system() == "Windows":
                        # Create and execute batch file method
                        batch_content = '''@echo off

cd /d D:
call mp-310\\Scripts\\activate.bat 
cmd /k'''
                        
                        with open('temp_venv_activate.bat', 'w') as f:
                            f.write(batch_content)
                        
                        subprocess.Popen(['start', 'temp_venv_activate.bat'], shell=True)
                except Exception as e2:
                    print(f"Fallback method also failed: {e2}")
            last_triggered_gesture = current_gesture
            last_gesture_timestamp = current_time

    # Reset last triggered gesture if current gesture is "none" and enough time has passed
    if (current_gesture == "none" and 
        last_triggered_gesture != "none" and
        (current_time - last_gesture_timestamp) > 2.0):  # 2 second reset time
        print("Gesture reset - ready for new commands")
        last_triggered_gesture = "none"

    # --- Visual Annotation for Display ---
    if result.gestures:
        for i, gesture_list in enumerate(result.gestures):
            category_name = gesture_list[0].category_name
            
            # Only draw recognized gestures that are in our allowed list
            if category_name not in allowed_gestures:
                continue
                
            score = round(gesture_list[0].score, 2)
            
            # Color coding based on confidence
            if score >= CONFIDENCE_THRESHOLD:
                text_color = (0, 255, 0)  # Green for high confidence
            else:
                text_color = (0, 0, 255)  # Red for low confidence
            
            hand_landmarks = result.hand_landmarks[i]
            
            # Position the text slightly above the wrist landmark
            title_x = int(hand_landmarks[0].x * annotated_image.shape[1])
            title_y = int(hand_landmarks[0].y * annotated_image.shape[0]) - 20
            
            # Display the gesture name and its confidence score
            cv2.putText(annotated_image, f"{category_name} ({score})", 
                        (title_x, title_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)

    # Draw the hand skeleton landmarks on the image
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2, circle_radius=2),
            )
            
    annotated_image_global = annotated_image

# Create the GestureRecognizer object
base_options = python.BaseOptions(model_asset_path='new.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=3,
    result_callback=print_and_draw_results
)

try:
    recognizer = vision.GestureRecognizer.create_from_options(options)
except Exception as e:
    print(f"Error creating GestureRecognizer: {e}")
    exit()

# Start capturing video from the DroidCam URL
#droidcam_url = 'http://10.xx.xx.xx:4747/video'
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(f"Error: Could not open video stream at {droidcam_url}")
    exit()

# Variable to keep track of the frame timestamp
frame_timestamp_ms = 0

print("Gesture Control System Started")
print("Available Commands (90% confidence required):")
print("C = Open Edge")
print("L = Maximize/Restore Current Window (Toggle)")
print("O = Open File Explorer")
print("W = Open YouTube in Chrome Guest Mode")
print("V = Open Command Prompt with Virtual Environment")
print("Press 'q' to quit")
print("-" * 50)

# Main processing loop
while cap.isOpened():
    try:
        # Read a frame from the video stream
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            time.sleep(0.01)
            continue

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize the global annotated image if it's the first time
        if annotated_image_global is None:
            annotated_image_global = np.copy(rgb_frame)

        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Get current timestamp
        frame_timestamp_ms = int(time.time() * 1000)

        # Perform gesture recognition asynchronously
        recognizer.recognize_async(mp_image, frame_timestamp_ms)

        # Convert annotated image back to BGR for display
        display_image = cv2.cvtColor(annotated_image_global, cv2.COLOR_RGB2BGR)

        # Add status information to the display
        status_text = f"Last Action: {last_triggered_gesture} | Confidence Req: {CONFIDENCE_THRESHOLD:.0%}"
        cv2.putText(display_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('ASL Gesture Control System', display_image)
    
        # Break on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if 'recognizer' in locals() and recognizer:
    recognizer.close()

# Clean up temporary files
try:
    import os
    temp_files = ['temp_activate.bat', 'temp_venv_activate.bat']
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
except:
    pass

print("Gesture Control System closed gracefully.")