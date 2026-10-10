
# Example 4.14 - Perbandingan ResNet50 dan DenseNet121
import cv2
import numpy as np
import threading
import time

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import resnet50, densenet
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# ==========================================
# 1. Memuat model
# ==========================================
print("Memuat ResNet50...")
resnet_model = resnet50.ResNet50(weights="imagenet")

print("Memuat DenseNet121...")
densenet_model = densenet.DenseNet121(weights="imagenet")

print("Kedua model berhasil dimuat.")
print("Pilih model: tekan 1 untuk ResNet50, 2 untuk DenseNet121.")

models = {
    "ResNet50": (resnet_model, resnet50.preprocess_input, 224),
    "DenseNet121": (densenet_model, densenet.preprocess_input, 224),
}

current_model_name = "ResNet50"

# ==========================================
# 2. Variabel hasil prediksi
# ==========================================
current_label = "Waiting..."
current_confidence = 0.0
prediction_time = 0.0
predicting = False

prediction_lock = threading.Lock()

# Statistik performa masing-masing model
stats = {
    "ResNet50": {"total_time": 0.0, "count": 0},
    "DenseNet121": {"total_time": 0.0, "count": 0},
}

# ==========================================
# 3. Fungsi prediksi di thread terpisah
# ==========================================
def predict_image(frame, model_name):
    global current_label, current_confidence
    global prediction_time, predicting

    try:
        model, preprocess, image_size = models[model_name]

        start_time = time.time()

        frame = cv2.resize(frame, (image_size, image_size))
        numpy_image = img_to_array(frame)
        image_batch = np.expand_dims(numpy_image, axis=0)

        # Preprocessing sesuai model yang dipilih
        processed_image = preprocess(image_batch.copy())

        predictions = model.predict(processed_image, verbose=0)
        results = decode_predictions(predictions, top=1)

        elapsed = time.time() - start_time

        with prediction_lock:
            current_label = results[0][0][1]
            current_confidence = results[0][0][2] * 100
            prediction_time = elapsed

            stats[model_name]["total_time"] += elapsed
            stats[model_name]["count"] += 1

    except Exception as error:
        print("Prediction error:", error)

    finally:
        predicting = False


# ==========================================
# 4. Membuka webcam
# ==========================================
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Kamera tidak dapat dibuka.")
    raise SystemExit

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==========================================
# 5. Proses webcam dan klasifikasi
# ==========================================
try:
    while True:
        ok, cam_frame = camera.read()

        if not ok:
            print("Gagal membaca frame kamera.")
            break

        # Mulai prediksi jika thread sebelumnya sudah selesai
        if not predicting:
            predicting = True
            selected_name = current_model_name

            thread = threading.Thread(
                target=predict_image,
                args=(cam_frame.copy(), selected_name),
                daemon=True
            )
            thread.start()

        with prediction_lock:
            label_text = (
                f"{current_model_name}: "
                f"{current_label} {current_confidence:.1f}%"
            )
            time_text = f"Last prediction: {prediction_time:.3f}s"

            count = stats[current_model_name]["count"]
            if count > 0:
                average_time = (
                    stats[current_model_name]["total_time"] / count
                )
            else:
                average_time = 0.0

            average_text = f"Average: {average_time:.3f}s ({count} predictions)"

        cv2.putText(
            cam_frame, label_text, (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2
        )
        cv2.putText(
            cam_frame, time_text, (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
        )
        cv2.putText(
            cam_frame, average_text, (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
        )
        cv2.putText(
            cam_frame, "1: ResNet50 | 2: DenseNet121 | ESC: Exit",
            (10, 125), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )

        cv2.imshow("Example 4.14 - Model Comparison", cam_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            current_model_name = "ResNet50"
            print("Model aktif: ResNet50")

        elif key == ord("2"):
            current_model_name = "DenseNet121"
            print("Model aktif: DenseNet121")

        elif key == 27:  # ESC
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

    print("\n===== HASIL PERBANDINGAN =====")

    for name, result in stats.items():
        count = result["count"]

        if count > 0:
            average = result["total_time"] / count
            print(f"{name}:")
            print(f"  Jumlah prediksi : {count}")
            print(f"  Rata-rata waktu : {average:.3f} detik")
        else:
            print(f"{name}: belum digunakan untuk prediksi.")
