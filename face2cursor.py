import warnings
import cv2
import mediapipe as mp
import pyautogui
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from scipy.ndimage import gaussian_filter1d
import time
import threading
import customtkinter
from concurrent.futures import ThreadPoolExecutor


customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

warnings.filterwarnings("ignore", category=UserWarning, module='google')

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 150)

if not cap.isOpened():
    print("Video source not available.")
    exit()

nose_positions = {'x': [], 'y': []}
blink_threshold = 0.15
blink_detected_left = False
blink_detected_right = False

window = customtkinter.CTk()
window.title("face2cursor :P")
window.geometry("1200x1000")

ui_texts = {
    "title": "face2cursor :P",
    "mirror": "Mirror Image",
    "sensitivity_x": "Sensitivity X",
    "sensitivity_y": "Sensitivity Y",
    "acceleration_x": "Acceleration X",
    "acceleration_y": "Acceleration Y",
    "left_eye": "Left Eye",
    "right_eye": "Right Eye",
    "preferred_eye": "Preferred Eye",
    "fps": "FPS",
    "nose_x": "Nose X",
    "nose_y": "Nose Y",
    "left_eye_ratio": "Left Eye Ratio",
    "right_eye_ratio": "Right Eye Ratio"
}

mirror_image = customtkinter.IntVar(value=1)
sensitivity_x = 1
sensitivity_y = 1
acceleration_x = 1
acceleration_y = 1
blink_eye = customtkinter.IntVar(value=1)  # 1: Left eye, 2: Right eye
last_update_time = time.time()

executor = ThreadPoolExecutor(max_workers=4)


def check_blink(left_eye_aspect_ratio, right_eye_aspect_ratio, blink_eye):
    global blink_detected_left, blink_detected_right

    if left_eye_aspect_ratio < blink_threshold and right_eye_aspect_ratio < blink_threshold:
        blink_detected_left = False
        blink_detected_right = False
    else:
        if blink_eye.get() == 1:
            if left_eye_aspect_ratio < blink_threshold and not blink_detected_left:
                blink_detected_left = True
                pyautogui.click()
            elif left_eye_aspect_ratio >= blink_threshold:
                blink_detected_left = False
        else:
            if right_eye_aspect_ratio < blink_threshold and not blink_detected_right:
                blink_detected_right = True
                pyautogui.click()
            elif right_eye_aspect_ratio >= blink_threshold:
                blink_detected_right = False


def update_sensitivity_x(val):
    global sensitivity_x
    sensitivity_x = float(val)
    lbl_sensitivity_x_value.configure(text=f'Sensitivity X = {val:.2f}')


def update_sensitivity_y(val):
    global sensitivity_y
    sensitivity_y = float(val)
    lbl_sensitivity_y_value.configure(text=f'Sensitivity Y = {val:.2f}')


def update_acceleration_x(val):
    global acceleration_x
    acceleration_x = float(val)
    lbl_acceleration_x_value.configure(text=f'Acceleration X = {val:.2f}')


def update_acceleration_y(val):
    global acceleration_y
    acceleration_y = float(val)
    lbl_acceleration_y_value.configure(text=f'Acceleration Y = {val:.2f}')


def convert_to_tk_image(cv_image):
    cv_image = cv2.resize(cv_image, (600, 500))
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    return ImageTk.PhotoImage(pil_image)


def calculate_eye_aspect_ratio(eye_landmarks, frame_width, frame_height):
    point = lambda i: np.array([eye_landmarks[i].x * frame_width, eye_landmarks[i].y * frame_height])
    horizontal_distance = np.linalg.norm(point(0) - point(3))
    vertical_distance_1 = np.linalg.norm(point(1) - point(5))
    vertical_distance_2 = np.linalg.norm(point(2) - point(4))
    return (vertical_distance_1 + vertical_distance_2) / (2.0 * horizontal_distance)


fps_limit = 50
prev_frame_time = 0


def draw_selected_landmarks(frame, face_landmarks, frame_width, frame_height):
    pass


fps = 0


def show_frame():
    global nose_positions, sensitivity_x, sensitivity_y, acceleration_x, acceleration_y, blink_eye, fps, cap, prev_frame_time, last_update_time

    ret, frame = cap.read()
    if not ret:
        print("Could not get frame.")
        return

    current_time = time.time()
    if current_time - prev_frame_time < 1.0 / fps_limit:
        window.after(10, show_frame)
        return

    prev_frame_time = current_time

    frame_height, frame_width, _ = frame.shape
    if mirror_image.get():
        frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[4]
            nose_x = int(nose_tip.x * frame_width)
            nose_y = int(nose_tip.y * frame_height)

            if len(nose_positions['x']) > 5:
                nose_positions['x'].pop(0)
                nose_positions['y'].pop(0)

            nose_positions['x'].append(nose_x)
            nose_positions['y'].append(nose_y)

            avg_nose_x = np.mean(gaussian_filter1d(nose_positions['x'], sigma=2))
            avg_nose_y = np.mean(gaussian_filter1d(nose_positions['y'], sigma=2))

            if len(nose_positions['x']) >= 3 and len(nose_positions['y']) >= 3:
                delta_x = (avg_nose_x - nose_x) * sensitivity_x * acceleration_x
                delta_y = (avg_nose_y - nose_y) * sensitivity_y * acceleration_y

                delta_x = -delta_x
                delta_y = -delta_y

                current_mouse_x, current_mouse_y = pyautogui.position()
                new_mouse_x = current_mouse_x + delta_x / 5
                new_mouse_y = current_mouse_y + delta_y / 5

                executor.submit(pyautogui.moveTo, new_mouse_x, new_mouse_y)

            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]

            left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
            right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]

            left_eye_aspect_ratio = calculate_eye_aspect_ratio(left_eye_landmarks, frame_width, frame_height)
            right_eye_aspect_ratio = calculate_eye_aspect_ratio(right_eye_landmarks, frame_width, frame_height)

            executor.submit(check_blink, left_eye_aspect_ratio, right_eye_aspect_ratio, blink_eye)
            executor.submit(draw_selected_landmarks, frame, face_landmarks, frame_width, frame_height)

            cv2.putText(frame, f'{ui_texts["left_eye_ratio"]}: {left_eye_aspect_ratio:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'{ui_texts["right_eye_ratio"]}: {right_eye_aspect_ratio:.2f}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'{ui_texts["nose_x"]}: {avg_nose_x:.2f}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'{ui_texts["nose_y"]}: {avg_nose_y:.2f}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            current_time = time.time()
            fps = 1 / (current_time - last_update_time)
            last_update_time = current_time

            cv2.putText(frame, f'{ui_texts["fps"]}: {fps:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    img_tk = convert_to_tk_image(frame)
    if img_tk:
        lbl_video.configure(image=img_tk)
        lbl_video.image = img_tk

    lbl_video.after(10, show_frame)


frame_controls = customtkinter.CTkFrame(window)
frame_controls.pack(side=customtkinter.TOP, pady=10)

lbl_video = tk.Label(window)
lbl_video.pack(pady=10)

chk_mirror = customtkinter.CTkCheckBox(frame_controls, text=ui_texts["mirror"], variable=mirror_image)
chk_mirror.pack(pady=5, padx=10)

# Sensitivity X
lbl_sensitivity_x_value = customtkinter.CTkLabel(frame_controls, text=f'Sensitivity X = {sensitivity_x:.2f}')
lbl_sensitivity_x_value.pack(side=customtkinter.TOP, pady=5)

sensitivity_slider_x = customtkinter.CTkSlider(frame_controls, from_=0.1, to=10.0, orientation="horizontal",
                                               number_of_steps=100, command=update_sensitivity_x)
sensitivity_slider_x.set(sensitivity_x)
sensitivity_slider_x.pack(side=customtkinter.TOP, pady=5, padx=10)

# Sensitivity Y
lbl_sensitivity_y_value = customtkinter.CTkLabel(frame_controls, text=f'Sensitivity Y = {sensitivity_y:.2f}')
lbl_sensitivity_y_value.pack(side=customtkinter.TOP, pady=5)

sensitivity_slider_y = customtkinter.CTkSlider(frame_controls, from_=0.1, to=10.0, orientation="horizontal",
                                               number_of_steps=100, command=update_sensitivity_y)
sensitivity_slider_y.set(sensitivity_y)
sensitivity_slider_y.pack(side=customtkinter.TOP, pady=5, padx=10)

# Acceleration X
lbl_acceleration_x_value = customtkinter.CTkLabel(frame_controls, text=f'Acceleration X = {acceleration_x:.2f}')
lbl_acceleration_x_value.pack(side=customtkinter.TOP, pady=5)

acceleration_slider_x = customtkinter.CTkSlider(frame_controls, from_=0.1, to=5.0, orientation="horizontal",
                                                number_of_steps=100, command=update_acceleration_x)
acceleration_slider_x.set(acceleration_x)
acceleration_slider_x.pack(side=customtkinter.TOP, pady=5, padx=10)

# Acceleration Y
lbl_acceleration_y_value = customtkinter.CTkLabel(frame_controls, text=f'Acceleration Y = {acceleration_y:.2f}')
lbl_acceleration_y_value.pack(side=customtkinter.TOP, pady=5)

acceleration_slider_y = customtkinter.CTkSlider(frame_controls, from_=0.1, to=5.0, orientation="horizontal",
                                                number_of_steps=100, command=update_acceleration_y)
acceleration_slider_y.set(acceleration_y)
acceleration_slider_y.pack(side=customtkinter.TOP, pady=5, padx=10)

lbl_preferred_eye = customtkinter.CTkLabel(frame_controls, text=ui_texts["preferred_eye"])
lbl_preferred_eye.pack(side=customtkinter.TOP, pady=5)

radio_btn_left_eye = customtkinter.CTkRadioButton(frame_controls, text=ui_texts["left_eye"], variable=blink_eye,
                                                  value=1)
radio_btn_left_eye.pack(side=customtkinter.LEFT, pady=5, padx=10)

radio_btn_right_eye = customtkinter.CTkRadioButton(frame_controls, text=ui_texts["right_eye"], variable=blink_eye,
                                                   value=2)
radio_btn_right_eye.pack(side=customtkinter.LEFT, pady=5, padx=10)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh:
    show_frame()
    window.mainloop()

cap.release()
cv2.destroyAllWindows()
