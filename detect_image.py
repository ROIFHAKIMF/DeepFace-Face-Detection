import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

import cv2
from deepface import DeepFace

# Path foto yang mau dideteksi
TARGET_IMG = "dataset/ROIF/ROIF9.jpg"  # Ganti sesuai kebutuhan

# Pilih backend detector:
# - 'opencv' / 'mediapipe' : Instan, super cepat
# - 'retinaface' / 'yolov8' : Presisi tinggi
DETECTOR = "centerface"  # Ganti sesuai kebutuhan: 'opencv', 'mediapipe', 'retinaface', 'yolov8'

def run_face_detection_only():
    if not os.path.exists(TARGET_IMG):
        print(f"[ERROR] File '{TARGET_IMG}' tidak ditemukan.")
        return

    try:
        print(f"[INFO] Memproses deteksi wajah pakai backend '{DETECTOR}'...")
        
        # Pure Detection: Hanya mengambil posisi wajah & confidence
        faces = DeepFace.extract_faces(
            img_path=TARGET_IMG,
            detector_backend=DETECTOR,
            enforce_detection=False
        )

        img = cv2.imread(TARGET_IMG)
        count = 0

        for face in faces:
            conf = face['confidence']
            
            # Filter hanya jika ada wajah yang terdeteksi
            if conf > 0:
                count += 1
                area = face['facial_area']
                x, y, w, h = area['x'], area['y'], area['w'], area['h']

                # Gambar Bounding Box Hijau
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"Face {count} ({conf:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(f"[SUCCESS] Terdeteksi {count} wajah.")
        cv2.imshow("Hasil Deteksi Wajah (Bounding Box Only)", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"[ERROR]: {e}")

if __name__ == "__main__":
    run_face_detection_only()