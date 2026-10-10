
# Example 4.14
# Real-Time Object Classification using ResNet50

import cv2
import numpy as np
import threading
import time

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications.imagenet_utils import decode_predictions


# =====================================================
# LOAD MODEL
# =====================================================

print("Loading ResNet50...")

model = resnet50.ResNet50(weights='imagenet')

print("ResNet50 berhasil dimuat.")

model.summary()


# =====================================================
# OPEN CAMERA
# =====================================================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Kamera tidak dapat dibuka.")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# =====================================================
# VARIABLES
# =====================================================

image_size = 224

current_label = "Waiting..."
current_confidence = 0.0
prediction_time = 0.0

predicting = False
prediction_lock = threading.Lock()


# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_image(frame):
    global current_label, current_confidence
    global prediction_time, predicting

    try:
        start_time = time.time()

        # Resize frame
        frame = cv2.resize(
            frame,
            (image_size, image_size)
        )

        # Convert image to array
        numpy_image = img_to_array(frame)

        # Add batch dimension
        image_batch = np.expand_dims(
            numpy_image,
            axis=0
        )

        # Preprocess for ResNet50
        processed_image = resnet50.preprocess_input(
            image_batch.copy()
        )

        # Predict
        predictions = model.predict(
            processed_image,
            verbose=0
        )

        # Decode prediction
        results = decode_predictions(
            predictions,
            top=1
        )

        with prediction_lock:
            current_label = results[0][0][1]
            current_confidence = results[0][0][2] * 100
            prediction_time = time.time() - start_time

    except Exception as error:
        print("Prediction error:", error)

    finally:
        predicting = False


# =====================================================
# CAMERA LOOP
# =====================================================

try:
    while True:
        ok, cam_frame = camera.read()

        if not ok:
            print("Gagal membaca frame kamera.")
            break

        # Start a prediction thread only when
        # the previous prediction has finished
        if not predicting:
            predicting = True

            thread = threading.Thread(
                target=predict_image,
                args=(cam_frame.copy(),),
                daemon=True
            )

            thread.start()

        # Read the latest prediction safely
        with prediction_lock:
            label_text = (
                f"ResNet50: {current_label} "
                f"{current_confidence:.1f}%"
            )

            time_text = (
                f"Prediction time: {prediction_time:.2f}s"
            )

        # Display prediction
        cv2.putText(
            cam_frame,
            label_text,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        cv2.putText(
            cam_frame,
            time_text,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Example 4.14 - ResNet50",
            cam_frame
        )

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
