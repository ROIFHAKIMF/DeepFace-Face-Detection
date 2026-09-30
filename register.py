import cv2
import os

def register_person():
    name = input("Masukkan nama orang (akan jadi nama folder): ").strip()
    if not name:
        print("Nama tidak boleh kosong!")
        return

    save_dir = os.path.join("dataset", name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    print("\n[INFO] Tekan 's' untuk simpan foto, tekan 'q' untuk selesai.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Register Dataset", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            count += 1
            file_path = os.path.join(save_dir, f"{name}{count}.jpg")
            cv2.imwrite(file_path, frame)
            print(f"Foto disimpan: {file_path}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_person()