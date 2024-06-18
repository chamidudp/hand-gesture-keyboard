

# Nose and Hand Gesture Controlled Virtual Keyboard

This project implements a virtual keyboard that can be controlled using nose gestures for selecting keys and right-hand gestures for pressing them. Utilizing Mediapipe for face and hand tracking, this innovative solution allows for hands-free typing, which can be particularly useful for individuals with limited mobility or for touchless interactions in hygienic environments.

## Features

- **Nose Gesture Selection**: Use your nose to hover over and select keys on the virtual keyboard.
- **Right Hand Gesture Press**: Use the index finger of your right hand to press the selected key.
- **Real-Time Feedback**: The virtual keyboard provides real-time visual feedback, highlighting selected keys.
- **Touchless Typing**: Enables hands-free typing, ideal for accessibility and hygienic use cases.

## Requirements

- Python 3.x
- Mediapipe
- OpenCV
- Pyperclip
- NumPy

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-username/nose-and-hand-gesture-keyboard.git
   cd nose-and-hand-gesture-keyboard
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Run the script**:
   ```sh
   python nose_hand_gesture_keyboard.py
   ```

2. **Interacting with the Keyboard**:
   - **Select a Key**: Move your nose over the desired key on the virtual keyboard to select it. The key will be highlighted.
   - **Press the Key**: Raise the index finger of your right hand to press the selected key. The typed word will appear below the keyboard.
   - **Copy Text**: Press the "COPY" button to copy the typed text to the clipboard.

3. **Exit the Application**: Press the 'q' key to exit the application.

## Project Structure

```
nose-and-hand-gesture-keyboard/
├── nose_hand_gesture_keyboard.py
├── requirements.txt
└── README.md
```

### requirements.txt

The `requirements.txt` file is used to specify the dependencies needed for the project. It lists the required Python packages along with their versions to ensure compatibility. Here is what it contains:

```
mediapipe==0.8.9.1
opencv-python==4.5.3.56
pyperclip==1.8.2
numpy==1.21.2
```

### Installing Dependencies

To install the dependencies listed in `requirements.txt`, run the following command:

```sh
pip install -r requirements.txt
```

This command will install all the necessary packages specified in the file.

## Code Explanation

### Import Libraries

- `os`, `cv2`, `mediapipe`, `pyperclip`, `time`, and `numpy` are imported for various functionalities like video capturing, drawing, and handling gestures.

### Suppress TensorFlow Lite Logging

- We suppress some of the logging output to keep the console clean.

### Initialize Mediapipe Face Mesh and Hands

- `mp_face_mesh` and `mp_hands` are used to detect face and hand landmarks.
- `mp_drawing` is used to draw landmarks on the screen.

### Variables

- `typed_word` holds the current text being typed.
- `cooldown_time` and `last_key_press_time` help control the typing speed.

### Function: `recognize_gesture`

- This function determines which key the nose is pointing at. It uses the nose's position to find the corresponding key on the virtual keyboard.

### Function: `draw_virtual_keyboard`

- This function draws the virtual keyboard on the screen. It highlights the selected key and shows the typed word.

### Function: `is_copy_button_pressed`

- This function checks if the copy button is pressed based on the nose's position.

### Main Code Block

- We capture video from the webcam.
- Initialize Mediapipe for face and hand detection.
- Continuously read frames from the webcam, process them, and display the virtual keyboard.
- The nose is used to select keys, and the right hand's index finger is used to press keys.

## Simple Explanation for Each Section

### Import Libraries

Think of these like getting different colored crayons before starting a drawing. Each library helps us do different things in our code.

### Suppress TensorFlow Lite Logging

This is like telling the noisy kids in the room to be quiet so you can focus on your work.

### Initialize Mediapipe Face Mesh and Hands

These are like the magic glasses that help us see and understand where the face and hands are.

### Variables

These are like little boxes where we store important information, like what we've typed so far.

### Function: `recognize_gesture`

This is a helper who looks at where your nose is pointing and tells us which key it is pointing at.

### Function: `draw_virtual_keyboard`

This helper draws the keyboard on the screen and shows which key is selected.

### Function: `is_copy_button_pressed`

This helper checks if the nose is over the "COPY" button.

### Main Code Block

This is the main part of our project. It continuously looks at the video from the webcam, finds where the nose and hands are, and updates the virtual keyboard.

## Contributions

Contributions are welcome! If you have any improvements or bug fixes, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Mediapipe](https://mediapipe.dev/) for providing powerful machine learning solutions for face and hand tracking.
- [OpenCV](https://opencv.org/) for computer vision functionalities.
- [Pyperclip](https://pyperclip.readthedocs.io/) for clipboard functionalities.
- [NumPy](https://numpy.org/) for numerical computing support.
```

You can use this content to create or update the `README.md` file in your GitHub repository. This version provides detailed explanations for each section of the code and offers simple explanations for easy understanding.
