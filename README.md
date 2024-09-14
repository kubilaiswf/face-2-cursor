# Facial Expression Based Cursor Control

This project implements a facial expression-based cursor control system using OpenCV, MediaPipe, and Tkinter. The system utilizes a webcam to track facial landmarks and control the mouse cursor based on nose movements. Additionally, it detects eye blinks to simulate mouse clicks. The interface allows for customizing sensitivity and acceleration of cursor movements.

## Features

- **Real-time Facial Landmark Detection**: Leverages MediaPipe for accurate facial landmark detection.
- **Nose-Based Cursor Control**: Moves the cursor based on the average position of the nose tip.
- **Eye Blink Detection**: Calculates eye aspect ratio to detect eye blinks and simulate mouse clicks.
- **Customizable Settings**: Provides controls for toggling mirror image, adjusting sensitivity and acceleration of cursor movements.
- **Graphical User Interface**: Built using Tkinter for ease of use.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/facial-expression-based-cursor-control.git
   cd facial-expression-based-cursor-control
   ```

2. **Set up the environment**

   Ensure you have Python 3.7 or later installed. Then install the required packages:

   ```bash
   pip install opencv-python mediapipe pyautogui pillow numpy scipy
   ```

## Usage

1. **Run the script**

   ```bash
   python main.py
   ```

2. **Interface Controls**

   - **Toggle Mirror Image**: Mirrors the webcam feed for more intuitive cursor control.
   - **Sensitivity Sliders**: Adjust sensitivity for both X and Y axes.
   - **Acceleration Sliders**: Adjust acceleration for both X and Y axes.

## How It Works

1. **Facial Landmark Detection**: Uses MediaPipe to detect and track facial landmarks in real-time.
2. **Nose Tracking**: Computes the average position of the nose tip and controls the cursor accordingly.
3. **Eye Aspect Ratio**: Calculates the aspect ratio of the eye to detect blinks. If the aspect ratio falls below a threshold, it simulates a mouse click.

## Explanation of Key Code Sections

### Eye Aspect Ratio Calculation

The function `calculate_eye_aspect_ratio` calculates the eye aspect ratio (EAR) to detect blinks:
```python
def calculate_eye_aspect_ratio(eye_landmarks, frame_width, frame_height):
    point = lambda i: np.array([eye_landmarks[i].x * frame_width, eye_landmarks[i].y * frame_height])
    ...
    return (vertical_distance_1 + vertical_distance_2) / (2.0 * horizontal_distance)
```

### Blink Detection
Blinks are detected by comparing the EAR with a predefined threshold:
```python
if left_eye_aspect_ratio < blink_threshold and not blink_detected:
    blink_detected = True
    pyautogui.click()
elif left_eye_aspect_ratio >= blink_threshold:
    blink_detected = False
```

## Dependencies

- **opencv-python**: For capturing and manipulating video streams.
- **mediapipe**: For detecting and tracking facial landmarks.
- **pyautogui**: For controlling the mouse cursor.
- **pillow**: For handling image operations in Tkinter.
- **numpy**: For numerical operations.
- **scipy**: For smoothing nose position data.

## Additional Information

- **Mirrored Image**: Toggle the mirrored image feature to flip the video feed horizontally for more intuitive control.
- **Adjustable Settings**: Use sliders to fine-tune cursor sensitivity and acceleration for precise control.

## Contributing

Feel free to submit issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any inquiries or support, please open an issue or contact [kubilay.karacar@gmail.com].
