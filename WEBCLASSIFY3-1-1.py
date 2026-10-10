
# Example 4.15
# Comparison of Image Classification Models using Webcam
# Models: VGG16, ResNet50, MobileNetV2, DenseNet201, InceptionV3

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


# ==================================================
# 1. Model configuration
# ==================================================

model_configs = {
    "vgg16": (VGG16, vgg16_preprocess, 224),
    "resnet50": (ResNet50, resnet50_preprocess, 224),
    "mobilenetv2": (MobileNetV2, mobilenetv2_preprocess, 224),
    "densenet201": (DenseNet201, densenet_preprocess, 224),
    "inceptionv3": (InceptionV3, inceptionv3_preprocess, 299),
}

model_names = list(model_configs.keys())

models = {}
preprocessors = {}
image_sizes = {}

# ==================================================
# 2. Load all models
# ==================================================

for name in model_names:
    print(f"\nLoading {name}...")

    ModelClass, preprocess_fn, size = model_configs[name]

    model = ModelClass(
        weights="imagenet",
        include_top=True,
        input_shape=(size, size, 3)
    )

    models[name] = model
    preprocessors[name] = preprocess_fn
    image_sizes[name] = size

    print(f"{name} loaded successfully.")

print("\nAll models loaded successfully!")


# ==================================================
# 3. Prediction variables
# ==================================================

current_model_index = 0

current_label = "Waiting..."
current_confidence = 0.0
prediction_time = 0.0
predicting = False

lock = threading.Lock()

stats = {
    name: {"total_time": 0.0, "count": 0}
    for name in model_names
}


# ==================================================
# 4. Prediction function
# ==================================================

def predict_image(frame, model_name):
    global current_label, current_confidence
    global prediction_time, predicting

    try:
        start_time = time.perf_counter()

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize image according to the selected model
        size = image_sizes[model_name]
        resized = cv2.resize(rgb_frame, (size, size))

        # Convert image to array and add batch dimension
        image = np.asarray(resized, dtype=np.float32)
        image = np.expand_dims(image, axis=0)

        # Apply model-specific preprocessing
        processed_image = preprocessors[model_name](image.copy())

        # Predict image class
        predictions = models[model_name].predict(
            processed_image,
            verbose=0
        )

        result = decode_predictions(predictions, top=1)[0][0]

        elapsed = time.perf_counter() - start_time

        # Update results and statistics
        with lock:
            current_label = result[1]
            current_confidence = result[2] * 100
            prediction_time = elapsed

            stats[model_name]["total_time"] += elapsed
            stats[model_name]["count"] += 1

    except Exception as error:
        print(f"Prediction error ({model_name}): {error}")

    finally:
        with lock:
            predicting = False


# ==================================================
# 5. Open webcam
# ==================================================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Cannot open webcam.")
    raise SystemExit

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ==================================================
# 6. Webcam classification loop
# ==================================================

try:
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Cannot read webcam frame.")
            break

        with lock:
            selected_model = model_names[current_model_index]

            if not predicting:
                predicting = True
                start_prediction = True
            else:
                start_prediction = False

            label = current_label
            confidence = current_confidence
            elapsed = prediction_time

            count = stats[selected_model]["count"]

            if count > 0:
                average_time = (
                    stats[selected_model]["total_time"] / count
                )
            else:
                average_time = 0.0

        # Run prediction in a background thread
        if start_prediction:
            threading.Thread(
                target=predict_image,
                args=(frame.copy(), selected_model),
                daemon=True
            ).start()

        # Display prediction
        cv2.putText(
            frame,
            f"{selected_model}: {label} ({confidence:.1f}%)",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Last prediction: {elapsed:.3f}s",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Average: {average_time:.3f}s | Samples: {count}",
            (10, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "1: VGG16  2: ResNet50  3: MobileNetV2",
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "4: DenseNet201  5: InceptionV3  ESC: Exit",
            (10, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        cv2.imshow("Example 4.15 - Model Comparison", frame)

        key = cv2.waitKey(1) & 0xFF

        # Switch model
        if ord("1") <= key <= ord("5"):
            new_index = key - ord("1")

            if new_index != current_model_index:
                current_model_index = new_index

                with lock:
                    current_label = "Waiting..."
                    current_confidence = 0.0
                    prediction_time = 0.0

                print(f"Selected model: {model_names[new_index]}")

        elif key == 27:
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

    # ==============================================
    # 7. Display performance comparison
    # ==============================================

    print("\n========== PERFORMANCE RESULTS ==========")
    print("Average prediction time (lower is faster)\n")

    for name in model_names:
        count = stats[name]["count"]

        if count > 0:
            average = stats[name]["total_time"] / count

            print(
                f"{name:15s} | "
                f"Predictions: {count:4d} | "
                f"Average: {average:.4f} seconds"
            )
        else:
            print(f"{name:15s} | No predictions recorded")

    print("\nProgram finished.")
