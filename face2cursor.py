import cv2
import mediapipe as mp
import pyautogui

# MediaPipe and OpenCV setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open video capture.")
    exit()

# Initial nose tip position
prev_nose_x = None
prev_nose_y = None

# Sensitivity multiplier (increase to make mouse move faster)
SENSITIVITY = 1.5  # Increase this value to make the cursor move faster

# Start the FaceMesh
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break

        # Mirror the image
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the face and get landmarks
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get the nose tip (landmark index: 1)
                nose_tip = face_landmarks.landmark[1]

                # Get frame dimensions
                frame_height, frame_width, _ = frame.shape

                # Calculate nose coordinates
                nose_x = int(nose_tip.x * frame_width)
                nose_y = int(nose_tip.y * frame_height)

                # Save the initial nose tip position
                if prev_nose_x is None and prev_nose_y is None:
                    prev_nose_x, prev_nose_y = nose_x, nose_y

                # Calculate movement (with sensitivity adjustment)
                delta_x = (nose_x - prev_nose_x) * SENSITIVITY
                delta_y = (nose_y - prev_nose_y) * SENSITIVITY

                # Get current mouse position
                current_mouse_x, current_mouse_y = pyautogui.position()

                # Move the mouse to the new position
                new_mouse_x = current_mouse_x + delta_x
                new_mouse_y = current_mouse_y + delta_y

                pyautogui.moveTo(new_mouse_x, new_mouse_y)

                # Update the previous nose position
                prev_nose_x, prev_nose_y = nose_x, nose_y

                # Optionally draw the face mesh (for debugging)
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1))

        # Remove the display window for full-screen support
        # cv2.imshow('face2cursor', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
