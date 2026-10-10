
# Example 4.15
# Image Classification using if-else to select models

import cv2
import numpy as np
import threading
import time

from tensorflow.keras.applications import (
    VGG16,
    ResNet50,
    MobileNetV2,
    DenseNet201,
    InceptionV3
)

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess
from tensorflow.keras.applications.imagenet_utils import decode_predictions


# ==========================================
# 1. Select model using if-else
# ==========================================

print("Select a model:")
print("1. VGG16")
print("2. ResNet50")
print("3. MobileNetV2")
print("4. DenseNet201")
print("5. InceptionV3")

choice = input("Enter model number (1-5): ")

if choice == "1":
    model_name = "VGG16"
    model_class = VGG16
    preprocess = vgg16_preprocess
    image_size = 224

elif choice == "2":
    model_name = "ResNet50"
    model_class = ResNet50
    preprocess = resnet50_preprocess
    image_size = 224

elif choice == "3":
    model_name = "MobileNetV2"
    model_class = MobileNetV2
    preprocess = mobilenetv2_preprocess
    image_size = 224

elif choice == "4":
    model_name = "DenseNet201"
    model_class = DenseNet201
    preprocess = densenet_preprocess
    image_size = 224

elif choice == "5":
    model_name = "InceptionV3"
    model_class = InceptionV3
    preprocess = inceptionv3_preprocess
    image_size = 299

else:
    print("Invalid choice. Default model: VGG16")
    model_name = "VGG16"
    model_class = VGG16
    preprocess = vgg16_preprocess
    image_size = 224


# ==========================================
# 2. Load selected model
# ==========================================

print(f"\nLoading {model_name}...")

model = model_class(
    weights="imagenet",
    include_top=True,
    input_shape=(image_size, image_size, 3)
)

print(f"{model_name} loaded successfully.")


# ==========================================
# 3. Open webcam
# ==========================================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Cannot open webcam.")
    raise SystemExit

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ==========================================
# 4. Prediction variables
# ==========================================

current_label = "Waiting..."
current_confidence = 0.0
prediction_time = 0.0
predicting = False

lock = threading.Lock()


# ==========================================
# 5. Prediction function
# ==========================================

def predict_image(frame):
    global current_label, current_confidence
    global prediction_time, predicting

    try:
        start_time = time.perf_counter()

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize image
        frame = cv2.resize(
            frame,
            (image_size, image_size)
        )

        # Convert to NumPy array
        image = np.asarray(frame, dtype=np.float32)
        image = np.expand_dims(image, axis=0)

        # Preprocess according to the selected model
        image = preprocess(image.copy())

        # Classify image
        predictions = model.predict(image, verbose=0)
        result = decode_predictions(predictions, top=1)[0][0]

        elapsed = time.perf_counter() - start_time

        with lock:
            current_label = result[1]
            current_confidence = result[2] * 100
            prediction_time = elapsed

    except Exception as error:
        print("Prediction error:", error)

    finally:
        with lock:
            predicting = False


# ==========================================
# 6. Webcam classification loop
# ==========================================

try:
    while True:
        ret, cam_frame = camera.read()

        if not ret:
            print("Cannot read webcam frame.")
            break

        with lock:
            start_prediction = not predicting

            if start_prediction:
                predicting = True

            label = current_label
            confidence = current_confidence
            elapsed = prediction_time

        if start_prediction:
            threading.Thread(
                target=predict_image,
                args=(cam_frame.copy(),),
                daemon=True
            ).start()

        # Display classification result
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
            f"Prediction time: {elapsed:.3f}s",
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

        cv2.imshow(
            f"Example 4.15 - {model_name}",
            cam_frame
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Program finished.")
