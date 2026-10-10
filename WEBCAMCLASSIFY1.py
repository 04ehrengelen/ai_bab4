# Example - VGG19 Webcam Classification

import cv2
import numpy as np
import threading
import time

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications.imagenet_utils import decode_predictions


# =========================================================
# LOAD VGG19
# =========================================================

print("Loading VGG19...")

start_time = time.time()

model = vgg19.VGG19(weights='imagenet')

load_time = time.time() - start_time

print("VGG19 berhasil dimuat.")
print(f"Waktu loading VGG19: {load_time:.2f} detik")


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Kamera tidak dapat dibuka.")
    exit()


camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# =========================================================
# VARIABLES
# =========================================================

image_size = 224

current_label = "Waiting..."

current_confidence = 0.0

predicting = False

running = True

prediction_time = 0.0


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(frame):

    global current_label
    global current_confidence
    global predicting
    global prediction_time

    try:

        # Start timer
        start = time.time()

        # Resize image
        image = cv2.resize(
            frame,
            (image_size, image_size)
        )

        # Convert image to array
        numpy_image = img_to_array(image)

        # Add batch dimension
        image_batch = np.expand_dims(
            numpy_image,
            axis=0
        )

        # Preprocess for VGG19
        processed_image = vgg19.preprocess_input(
            image_batch
        )

        # Prediction
        predictions = model.predict(
            processed_image,
            verbose=0
        )

        # Get best prediction
        results = decode_predictions(
            predictions,
            top=1
        )

        current_label = results[0][0][1]

        current_confidence = results[0][0][2] * 100

        # Calculate prediction time
        prediction_time = time.time() - start

    except Exception as e:

        print("Prediction error:", e)

    finally:

        predicting = False


# =========================================================
# CAMERA LOOP
# =========================================================

while running:

    ok, cam_frame = camera.read()

    if not ok:
        print("Gagal membaca frame.")
        break


    # =====================================================
    # START PREDICTION
    # =====================================================

    if not predicting:

        predicting = True

        prediction_thread = threading.Thread(
            target=predict_image,
            args=(cam_frame.copy(),)
        )

        prediction_thread.start()


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    text = (
        f"VGG19: {current_label} "
        f"{current_confidence:.1f}%"
    )

    cv2.putText(
        cam_frame,
        text,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )


    # Display prediction time
    time_text = f"Prediction time: {prediction_time:.2f}s"

    cv2.putText(
        cam_frame,
        time_text,
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # =====================================================
    # SHOW CAMERA
    # =====================================================

    cv2.imshow(
        "VGG19 Real-Time Classification",
        cam_frame
    )


    # =====================================================
    # ESC TO EXIT
    # =====================================================

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        running = False


# =========================================================
# RELEASE CAMERA
# =========================================================

camera.release()

cv2.destroyAllWindows()