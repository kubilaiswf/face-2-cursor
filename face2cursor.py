import cv2
import mediapipe as mp
import pyautogui
import time

# MediaPipe ve OpenCV ayarları
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Ekran çözünürlüğünü alıyoruz
screen_width, screen_height = pyautogui.size()

# Video dosyasını alıyoruz
video_path = 'videos/ornek-video.mp4'  # Video dosyanızın yolunu girin
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video dosyası açılamadı.")
    exit()

# Fareyi manuel ve burunla kontrol etmek için son fare konumu
prev_mouse_x, prev_mouse_y = pyautogui.position()
last_move_time = time.time()

# FaceMesh'i başlatıyoruz
with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # Yalnızca bir yüzü takip edeceğiz
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()  # Videodan bir kare okuyoruz
        if not ret:
            print("Video bitti.")
            break

        # Video çözünürlüğünü düşürerek performansı artırabiliriz
        frame = cv2.resize(frame, (640, 360))

        # OpenCV için BGR'den RGB'ye dönüşüm
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Yüzü işleyip anahtar noktaları alıyoruz
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Burun ucunun konumunu alıyoruz (landmark index: 1)
                nose_tip = face_landmarks.landmark[1]

                # Kare boyutlarını alıyoruz
                frame_height, frame_width, _ = frame.shape

                # Burun ucunun X ve Y koordinatlarını çerçeveye göre hesaplıyoruz
                nose_x = int(nose_tip.x * frame_width)
                nose_y = int(nose_tip.y * frame_height)

                # Fareyi ekrandaki konuma göre hareket ettiriyoruz (her 0.1 saniyede bir)
                if time.time() - last_move_time > 0.1:
                    screen_x = screen_width * (nose_x / frame_width)
                    screen_y = screen_height * (nose_y / frame_height)

                    # Farenin hareket etmesine izin verelim, ancak küçük aralıklarla
                    pyautogui.moveTo(screen_x, screen_y)
                    last_move_time = time.time()

                # Anahtar noktaları ve çizgileri küçültüyoruz
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),  # Küçük noktalar
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1))  # İnce çizgiler

        # Kameradan gelen görüntüyü ekranda gösteriyoruz
        cv2.imshow('Yüz Takip', frame)

        # 'q' tuşuna basılınca döngüyü sonlandır
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Kaynakları serbest bırakıyoruz
cap.release()
cv2.destroyAllWindows()
