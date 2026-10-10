
# Example 4.15 - Webcam Classification tanpa membekukan tampilan

import cv2
import numpy as np
import threading
import time

from classification_models.keras import Classifiers
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# ==========================================
# 1. Pilih model
# ==========================================
model_name = "vgg16"
# model_name = "resnet50"
# model_name = "mobilenetv2"
# model_name = "densenet201"
# model_name = "inceptionv3"

image_size = 299 if model_name == "inceptionv3" else 224

# ==========================================
# 2. Memuat model
# ==========================================
print(f"Memuat model {model_name}...")
Classifier, preprocess_input = Classifiers.get(model_name)

model = Classifier(
    input_shape=(image_size, image_size, 3),
    weights="imagenet",
    classes=1000
)

print("Model berhasil dimuat!")

# ==========================================
# 3. Variabel hasil prediksi
# ==========================================
current_label = "Menunggu prediksi..."
current_confidence = 0.0
prediction_time = 0.0
predicting = False

result_lock = threading.Lock()

# ==========================================
# 4. Prediksi di thread terpisah
# ==========================================
def predict_image(frame):
    global current_label, current_confidence
    global prediction_time, predicting

    try:
        start_time = time.time()

        # BGR ke RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize dan ubah ke array
        frame = cv2.resize(frame, (image_size, image_size))
        image = np.asarray(frame, dtype=np.float32)
        image = np.expand_dims(image, axis=0)

        # Preprocessing sesuai model
        image = preprocess_input(image.copy())

        # Prediksi
        predictions = model.predict(image, verbose=0)
        results = decode_predictions(predictions, top=1)

        label = results[0][0][1]
        confidence = results[0][0][2] * 100
        elapsed = time.time() - start_time

        with result_lock:
            current_label = label
            current_confidence = confidence
            prediction_time = elapsed

    except Exception as error:
        print("Prediction error:", error)

    finally:
        with result_lock:
            predicting = False

# ==========================================
# 5. Membuka webcam
# ==========================================
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Kamera tidak dapat dibuka.")
    raise SystemExit

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==========================================
# 6. Menampilkan webcam
# ==========================================
try:
    while True:
        ret, cam_frame = camera.read()

        if not ret:
            print("Gagal membaca frame webcam.")
            break

        # Mulai prediksi jika tidak ada prediksi yang berjalan
        with result_lock:
            should_predict = not predicting

            if should_predict:
                predicting = True

            label = current_label
            confidence = current_confidence
            elapsed = prediction_time

        if should_predict:
            threading.Thread(
                target=predict_image,
                args=(cam_frame.copy(),),
                daemon=True
            ).start()

        # Tampilkan hasil prediksi terakhir
        cv2.putText(
            cam_frame,
            f"{model_name}: {label} ({confidence:.1f}%)",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            cam_frame,
            f"Prediction time: {elapsed:.2f}s",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            cam_frame,
            "Press ESC to exit",
            (10, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Example 4.15 - " + model_name, cam_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Webcam ditutup.")
