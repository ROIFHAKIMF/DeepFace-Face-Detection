import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

import cv2
from deepface import DeepFace

# Pilihan detector backend:
# - 'opencv'    : Super cepat, paling ringan untuk webcam (Rekomendasi)
# - 'mediapipe' : Sangat lancar, responsif & presisi
# - 'yolov8'    : Cepat & bagus untuk deteksi banyak wajah
DETECTOR = "centerface"  # Ganti sesuai kebutuhan: 'opencv', 'mediapipe', 'retinaface', 'yolov8'

def run_webcam_detection_only():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat diakses.")
        return

    print(f"[INFO] Menjalankan webcam dengan backend '{DETECTOR}'...")
    print("[INFO] Tekan tombol 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal menerima frame dari webcam.")
            break

        try:
            # Deteksi posisi wajah pada frame saat ini
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=DETECTOR,
                enforce_detection=False
            )

            count = 0
            for face in faces:
                conf = face['confidence']
                
                # Filter jika confidence/keyakinan deteksi cukup tinggi
                if conf > 0.5:
                    count += 1
                    area = face['facial_area']
                    x, y, w, h = area['x'], area['y'], area['w'], area['h']

                    # Gambar Bounding Box Hijau & Label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Wajah {count} ({conf:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Tampilkan jumlah wajah terdeteksi di pojok kiri atas
            cv2.putText(frame, f"Total Wajah: {count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        except Exception:
            pass

        cv2.imshow("Real-Time Face Detection (Bounding Box)", frame)

        # Tekan 'q' pada keyboard untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_detection_only()