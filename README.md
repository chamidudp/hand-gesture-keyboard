Hand Gesture Controlled Interactive Painting
This project implements an interactive painting application where users can paint on a virtual canvas using hand gestures detected via OpenCV and Mediapipe. It provides a unique and intuitive way to create digital art, leveraging real-time hand tracking and gesture recognition.

Features
Gesture-Based Painting: Use hand gestures to draw lines and shapes on a virtual canvas.
Real-Time Feedback: Visual indicators show the current drawing tool, color, and canvas position.
Customizable Brushes: Select from various brush sizes and colors using intuitive hand movements.
Creative Freedom: Enables users to paint digitally without physical tools, enhancing creativity and accessibility.
Requirements
Python 3.x
OpenCV
Mediapipe
NumPy
Installation
Clone the repository:

sh
Copy code
git clone https://github.com/your-username/gesture-controlled-painting.git
cd gesture-controlled-painting
Set up a virtual environment (optional but recommended):

sh
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:

sh
Copy code
pip install -r requirements.txt
Usage
Run the script:

sh
Copy code
python gesture_controlled_painting.py
Interacting with the Painting Application:

Draw on Canvas: Use your index finger to draw lines and shapes on the virtual canvas.
Change Brush Size: Move your hand closer or further from the camera to adjust the brush size.
Select Color: Use gestures to choose different colors for painting.
Undo/Redo: Perform specific gestures (e.g., swipe left/right) to undo or redo actions.
Clear Canvas: Use a specific gesture (e.g., making a fist) to clear the entire canvas.
Exit the Application: Press the 'q' key to exit the painting application.

Project Structure
Copy code
gesture-controlled-painting/
├── gesture_controlled_painting.py
├── requirements.txt
└── README.md
requirements.txt
makefile
Copy code
opencv-python==4.5.3.56
mediapipe==0.8.9.1
numpy==1.21.2
Installing Dependencies
To install the dependencies listed in requirements.txt, run the following command:

sh
Copy code
pip install -r requirements.txt
This command installs all necessary packages for the project.

Code Explanation
Import Libraries
python
Copy code
import cv2
import mediapipe as mp
import numpy as np
cv2: OpenCV library for computer vision tasks, including image and video processing.
mediapipe: Google's Mediapipe library for hand tracking.
numpy: Library for numerical computing and array operations.
Initialize Mediapipe Hands
python
Copy code
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
Initializes Mediapipe for hand tracking (mp_hands) and drawing utilities (mp_drawing).
Global Variables
python
Copy code
gesture_start_time = 0
gesture_duration = 1.5
gesture_start_time: Tracks the time when a gesture is recognized.
gesture_duration: Maximum time duration considered for a continuous gesture.
Function: detect_gesture
python
Copy code
def detect_gesture(hand_landmarks):
    # Detects and interprets hand gestures to control the painting application
Analyzes hand landmarks to interpret gestures like drawing, changing brush size, selecting colors, and managing canvas actions.
Function: draw_interface
python
Copy code
def draw_interface(image, current_tool, current_color):
    # Draws the painting interface with visual feedback for detected gestures
Renders the virtual canvas, current drawing tool, selected color, and visual feedback for detected gestures on the screen.
Main Program Execution
python
Copy code
# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe hands model
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)  # Flip image horizontally for a mirror effect

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image color to RGB
        results = hands.process(image_rgb)  # Process image for hand detection

        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Convert image color back to BGR for OpenCV

        # Extract hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture = detect_gesture(hand_landmarks)

                # Perform actions based on detected gesture
                if gesture == "DRAW":
                    # Draw on canvas
                elif gesture == "CHANGE_BRUSH_SIZE":
                    # Adjust brush size
                elif gesture == "SELECT_COLOR":
                    # Choose color
                elif gesture == "UNDO":
                    # Undo last action
                elif gesture == "REDO":
                    # Redo last undone action
                elif gesture == "CLEAR_CANVAS":
                    # Clear entire canvas

        # Draw painting interface with canvas and gesture feedback
        draw_interface(image, current_tool, current_color)

        cv2.imshow('Gesture-Controlled Painting', image)  # Display the painting interface

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop on 'q' key press
            break

cap.release()
cv2.destroyAllWindows()
Explanation
Capture Video: Initializes webcam capture (cap) using OpenCV.
Main Loop: Continuously reads frames from the webcam, processes them for hand detection using Mediapipe, and updates the painting interface.
Hand Detection: Uses Mediapipe to detect and track hand landmarks for gesture recognition.
Gesture Recognition: Interprets hand gestures to control actions like drawing, adjusting brush size, selecting colors, and managing canvas actions (undo, redo, clear).
Interface Rendering: Draws the virtual canvas, current drawing tool, selected color, and visual feedback for detected gestures on the screen.
This project enables interactive and creative painting using hand gestures, suitable for applications requiring touchless interaction, digital art creation, and enhanced accessibility. Adjustments and enhancements can be made to customize gesture recognition or add additional functionalities as needed.
