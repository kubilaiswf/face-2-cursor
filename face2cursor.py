import cv2
import mediapipe as mp
import pyautogui

# MediaPipe ve OpenCV ayarları
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Video dosyasını alıyoruz (webcam kullanıyoruz)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Video dosyası açılamadı.")
    exit()

# Burun ucu başlangıç değerleri
prev_nose_x = None
prev_nose_y = None

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

        # Görüntüyü aynalamak için flip işlemi yapıyoruz
        frame = cv2.flip(frame, 1)  # Yatay eksende görüntüyü çeviriyoruz

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

                # İlk çerçeve için burun ucunun başlangıç pozisyonunu kaydediyoruz
                if prev_nose_x is None and prev_nose_y is None:
                    prev_nose_x, prev_nose_y = nose_x, nose_y

                # Burnun hareketine göre fareyi mevcut konumundan hareket ettiriyoruz
                delta_x = nose_x - prev_nose_x
                delta_y = nose_y - prev_nose_y

                # Şu anki fare konumunu alıyoruz
                current_mouse_x, current_mouse_y = pyautogui.position()

                # Fareyi göreceli hareket ettiriyoruz (mevcut konuma delta ekliyoruz)
                new_mouse_x = current_mouse_x + delta_x
                new_mouse_y = current_mouse_y + delta_y

                # Fareyi yeni pozisyona taşı
                pyautogui.moveTo(new_mouse_x, new_mouse_y)

                # Burun ucunun önceki pozisyonlarını güncelliyoruz
                prev_nose_x, prev_nose_y = nose_x, nose_y

                # İsteğe bağlı: Anahtar noktayı ekrana çiziyoruz (daha küçük boyutlar)
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),  # Küçük noktalar
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1))  # İnce çizgiler

        # Kameradan gelen görüntüyü ekranda gösteriyoruz
        cv2.imshow('face2cursor', frame)

        # 'q' tuşuna basılınca döngüyü sonlandır
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Kaynakları serbest bırakıyoruz
cap.release()
cv2.destroyAllWindows()
