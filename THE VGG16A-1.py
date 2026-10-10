# Compare VGG16 and VGG19
# Example 4.11

import numpy as np
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input as preprocess_vgg16
)

from tensorflow.keras.applications.vgg19 import (
    VGG19,
    preprocess_input as preprocess_vgg19
)

from tensorflow.keras.applications.imagenet_utils import decode_predictions


# =========================================================
# Lokasi gambar
# =========================================================

img_path = r'D:\KAMPUS\SEMESTER 5\COMPUTER VISION\AI\image\Optimus Prime.jpg'


# =========================================================
# Load gambar
# =========================================================

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)


# =========================================================
# VGG16
# =========================================================

print("\n================ VGG16 ================")

model_vgg16 = VGG16(weights='imagenet')

x_vgg16 = preprocess_vgg16(x.copy())

predictions_vgg16 = model_vgg16.predict(x_vgg16)

results_vgg16 = decode_predictions(
    predictions_vgg16,
    top=5
)

print("\nTop 5 Prediction VGG16:")

for rank, (_, class_name, score) in enumerate(results_vgg16[0], start=1):
    print(f"{rank}. {class_name}: {score * 100:.2f}%")


# =========================================================
# VGG19
# =========================================================

print("\n================ VGG19 ================")

model_vgg19 = VGG19(weights='imagenet')

x_vgg19 = preprocess_vgg19(x.copy())

predictions_vgg19 = model_vgg19.predict(x_vgg19)

results_vgg19 = decode_predictions(
    predictions_vgg19,
    top=5
)

print("\nTop 5 Prediction VGG19:")

for rank, (_, class_name, score) in enumerate(results_vgg19[0], start=1):
    print(f"{rank}. {class_name}: {score * 100:.2f}%")