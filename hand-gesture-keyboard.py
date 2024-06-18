import os
import cv2
import mediapipe as mp # type: ignore
import pyperclip
import time

# Suppress TensorFlow Lite logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Mediapipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

typed_word = ""
cooldown_time = 0.5  # Time in seconds to wait before allowing another letter to be added
last_key_press_time = 0

key_width, key_height = 60, 60  # Original smaller key size

def recognize_gesture(finger_tip_x, finger_tip_y, start_x, start_y):
    if start_x <= finger_tip_x < start_x + key_width * 10 and start_y <= finger_tip_y < start_y + key_height * 3:
        col = (finger_tip_x - start_x) // key_width
        row = (finger_tip_y - start_y) // key_height
        keys = "QWERTYUIOPASDFGHJKLZXCVBNM"
        index = row * 10 + col
        if index < len(keys):
            key = keys[index]
            return key
    # Check if the space or delete key is pressed
    if start_x + key_width * 4 <= finger_tip_x < start_x + key_width * 6 and start_y + key_height * 3 <= finger_tip_y < start_y + key_height * 4:
        return " "
    if start_x + key_width * 9 <= finger_tip_x < start_x + key_width * 10 and start_y + key_height * 3 <= finger_tip_y < start_y + key_height * 4:
        return "DELETE"
    return None

# Function to draw the virtual keyboard
def draw_virtual_keyboard(image, start_x, start_y, highlighted_key=None, typed_word=""):
    keys = "QWERTYUIOP\nASDFGHJKL\nZXCVBNM"
    for i, row in enumerate(keys.split('\n')):
        for j, key in enumerate(row):
            x = start_x + j * key_width
            y = start_y + i * key_height
            if key == highlighted_key:
                cv2.rectangle(image, (x, y), (x + key_width, y + key_height), (0, 255, 0), -1)
            else:
                cv2.rectangle(image, (x, y), (x + key_width, y + key_height), (255, 0, 0), 2)
            cv2.putText(image, key, (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Draw the space key
    x = start_x + key_width * 4
    y = start_y + key_height * 3
    if highlighted_key == " ":
        cv2.rectangle(image, (x, y), (x + key_width * 2, y + key_height), (0, 255, 0), -1)
    else:
        cv2.rectangle(image, (x, y), (x + key_width * 2, y + key_height), (255, 0, 0), 2)
    cv2.putText(image, "SPACE", (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Draw the delete key
    x = start_x + key_width * 9
    if highlighted_key == "DELETE":
        cv2.rectangle(image, (x, y), (x + key_width, y + key_height), (0, 255, 0), -1)
    else:
        cv2.rectangle(image, (x, y), (x + key_width, y + key_height), (255, 0, 0), 2)
    cv2.putText(image, "DEL", (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the typed word
    cv2.putText(image, typed_word, (start_x, start_y + 4 * key_height + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(image, (start_x, start_y + 4 * key_height + 60), (start_x + 150, start_y + 4 * key_height + 110), (0, 0, 255), 2)
    cv2.putText(image, "COPY", (start_x + 30, start_y + 4 * key_height + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Function to check if the copy button is pressed
def is_copy_button_pressed(finger_tip_x, finger_tip_y, start_x, start_y):
    if start_x <= finger_tip_x <= start_x + 150 and start_y + 4 * key_height + 60 <= finger_tip_y <= start_y + 4 * key_height + 110:
        return True
    return False

# Capture video from webcam
cap = cv2.VideoCapture(0)

selected_key = None
with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Flip the image horizontally for a mirror effect
        image = cv2.flip(image, 1)

        # Convert the image color to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(image_rgb)

        # Convert the image color back to BGR for OpenCV
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Get image dimensions
        image_height, image_width, _ = image.shape

        # Calculate the starting position of the keyboard to center it
        start_x = (image_width - key_width * 10) // 2
        start_y = (image_height - key_height * 4) // 2

        # Create a copy of the image for displaying the virtual keyboard
        keyboard_image = image.copy()

        right_hand_landmarks = None

        # Extract right hand landmarks
        if results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                if handedness.classification[0].label == "Right":
                    right_hand_landmarks = hand_landmarks

        # Recognize gesture from right hand to select key
        if right_hand_landmarks:
            landmarks = right_hand_landmarks.landmark
            finger_tip_x = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
            finger_tip_y = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)

            selected_key = recognize_gesture(finger_tip_x, finger_tip_y, start_x, start_y)

            # Recognize gesture from right hand to press key or copy button
            index_finger_tip_y = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
            index_finger_mcp_y = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)

            if index_finger_tip_y < index_finger_mcp_y and time.time() - last_key_press_time > cooldown_time:
                if selected_key:
                    if selected_key == "DELETE":
                        typed_word = typed_word[:-1]
                    else:
                        typed_word += selected_key
                    selected_key = None
                    last_key_press_time = time.time()
            
            # Check if copy button is pressed
            if is_copy_button_pressed(finger_tip_x, finger_tip_y, start_x, start_y):
                pyperclip.copy(typed_word)
                last_key_press_time = time.time()  # Add cooldown to avoid multiple copies

        draw_virtual_keyboard(keyboard_image, start_x, start_y, selected_key, typed_word)

        # Display the resulting frames
        cv2.imshow('Virtual Keyboard', keyboard_image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
