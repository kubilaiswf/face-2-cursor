import warnings
import cv2
import mediapipe as mp
import pyautogui
import tkinter as tk
from tkinter import Button, Label, Frame, Scale, HORIZONTAL
from PIL import Image, ImageTk
import numpy as np
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore", category=UserWarning, module='google')

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

if not cap.isOpened():
    print("Video kaynağı açılamadı.")
    exit()

mirror_image = False
sensitivity_x = 1
sensitivity_y = 1
acceleration_x = 1
acceleration_y = 1
nose_positions = {'x': [], 'y': []}
blink_threshold = 0.2
blink_detected = False


def toggle_mirror():
    global mirror_image, nose_positions
    mirror_image = not mirror_image
    nose_positions = {'x': [], 'y': []}


def update_sensitivity_x(val):
    global sensitivity_x
    sensitivity_x = float(val)


def update_sensitivity_y(val):
    global sensitivity_y
    sensitivity_y = float(val)


def update_acceleration_x(val):
    global acceleration_x
    acceleration_x = float(val)


def update_acceleration_y(val):
    global acceleration_y
    acceleration_y = float(val)


window = tk.Tk()
window.title("Yüz Takip")
window.geometry("800x600")


def convert_to_tk_image(cv_image):
    cv_image = cv2.resize(cv_image, (800, 600))
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    return ImageTk.PhotoImage(pil_image)


def calculate_eye_aspect_ratio(eye_landmarks, frame_width, frame_height):
    point = lambda i: np.array([eye_landmarks[i].x * frame_width, eye_landmarks[i].y * frame_height])

    horizontal_distance = np.linalg.norm(point(0) - point(3))
    vertical_distance_1 = np.linalg.norm(point(1) - point(5))
    vertical_distance_2 = np.linalg.norm(point(2) - point(4))

    return (vertical_distance_1 + vertical_distance_2) / (2.0 * horizontal_distance)


def show_frame():
    global nose_positions, sensitivity_x, sensitivity_y, acceleration_x, acceleration_y, blink_detected

    ret, frame = cap.read()
    if not ret:
        print("Kamera verisi alınamıyor.")
        return

    frame_height, frame_width, _ = frame.shape

    if mirror_image:
        frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]
            nose_x = int(nose_tip.x * frame_width)
            nose_y = int(nose_tip.y * frame_height)

            nose_positions['x'].append(nose_x)
            nose_positions['y'].append(nose_y)

            if len(nose_positions['x']) > 5:
                nose_positions['x'].pop(0)
                nose_positions['y'].pop(0)

            avg_nose_x = np.mean(gaussian_filter1d(nose_positions['x'], sigma=2))
            avg_nose_y = np.mean(gaussian_filter1d(nose_positions['y'], sigma=2))

            delta_x = (avg_nose_x - nose_x) * sensitivity_x * acceleration_x
            delta_y = (avg_nose_y - nose_y) * sensitivity_y * acceleration_y

            delta_x = -delta_x
            delta_y = -delta_y

            current_mouse_x, current_mouse_y = pyautogui.position()
            new_mouse_x = current_mouse_x + delta_x / 5
            new_mouse_y = current_mouse_y + delta_y / 5

            pyautogui.moveTo(new_mouse_x, new_mouse_y)

            # Sol göz kırpma tespiti için doğru landmark indeksleri kullanılarak göz oranını hesapla
            left_eye_landmarks = [
                face_landmarks.landmark[362], face_landmarks.landmark[385], face_landmarks.landmark[387],
                face_landmarks.landmark[263], face_landmarks.landmark[373], face_landmarks.landmark[380]
            ]
            left_eye_aspect_ratio = calculate_eye_aspect_ratio(left_eye_landmarks, frame_width, frame_height)

            print(f'Sol Göz Kırpma Oranı: {left_eye_aspect_ratio:.2f}')
            cv2.putText(frame, f'Sol Goz Orani: {left_eye_aspect_ratio:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            if left_eye_aspect_ratio < blink_threshold and not blink_detected:
                blink_detected = True
                pyautogui.click()
            elif left_eye_aspect_ratio >= blink_threshold:
                blink_detected = False

            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1))

    img_tk = convert_to_tk_image(frame)
    lbl_video.imgtk = img_tk
    lbl_video.configure(image=img_tk)

    lbl_video.after(10, show_frame)


frame_controls = Frame(window, bd=2, relief=tk.SUNKEN)
frame_controls.pack(side=tk.TOP, pady=10)

lbl_video = Label(window)
lbl_video.pack(pady=10)

btn_toggle = Button(frame_controls, text="Görüntüyü Aynala", command=toggle_mirror, bg="lightblue",
                    font=("Helvetica", 14))
btn_toggle.pack(pady=5, padx=10)

sensitivity_slider_x = Scale(frame_controls, from_=0.1, to=5.0, orient=HORIZONTAL, resolution=0.1, label="Hassasiyet X",
                             command=update_sensitivity_x)
sensitivity_slider_x.set(sensitivity_x)
sensitivity_slider_x.pack(side=tk.LEFT, pady=5, padx=10)

sensitivity_slider_y = Scale(frame_controls, from_=0.1, to=5.0, orient=HORIZONTAL, resolution=0.1, label="Hassasiyet Y",
                             command=update_sensitivity_y)
sensitivity_slider_y.set(sensitivity_y)
sensitivity_slider_y.pack(side=tk.LEFT, pady=5, padx=10)

acceleration_slider_x = Scale(frame_controls, from_=0.1, to=5.0, orient=HORIZONTAL, resolution=0.1, label="İvme X",
                              command=update_acceleration_x)
acceleration_slider_x.set(acceleration_x)
acceleration_slider_x.pack(side=tk.LEFT, pady=5, padx=10)

acceleration_slider_y = Scale(frame_controls, from_=0.1, to=5.0, orient=HORIZONTAL, resolution=0.1, label="İvme Y",
                              command=update_acceleration_y)
acceleration_slider_y.set(acceleration_y)
acceleration_slider_y.pack(side=tk.LEFT, pady=5, padx=10)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    show_frame()
    window.mainloop()

cap.release()
cv2.destroyAllWindows()
