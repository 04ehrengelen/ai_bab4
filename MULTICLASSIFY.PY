# Example 4.12

import numpy as np

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.imagenet_utils import decode_predictions

from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3
)
# =========================================================
# INIT THE MODELS
# =========================================================
# Pilih salah satu model
# model = vgg16.VGG16(weights='imagenet')
# model = resnet50.ResNet50(weights='imagenet')
model = mobilenet.MobileNet(weights='imagenet')
# model = inception_v3.InceptionV3(weights='imagenet')
# =========================================================
# MODEL SUMMARY
# =========================================================
model.summary()
# =========================================================
# LOAD IMAGE
# =========================================================

img_path = r'D:\KAMPUS\SEMESTER 5\COMPUTER VISION\AI\image\Optimus Prime.jpg'

img = image.load_img(
    img_path,
    target_size=(224, 224)
)
# =========================================================
# CONVERT IMAGE TO ARRAY
# ========================================================
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# =========================================================
# PREPROCESS IMAGE
# =========================================================
# Untuk VGG16
# processed_image = vgg16.preprocess_input(x)
# Untuk ResNet50
# processed_image = resnet50.preprocess_input(x)
# Untuk MobileNet
processed_image = mobilenet.preprocess_input(x)
# Untuk InceptionV3
# processed_image = inception_v3.preprocess_input(x)
# =========================================================
# PREDICTION
# =========================================================
predictions = model.predict(processed_image)
# =========================================================
# DECODE PREDICTIONS
# =========================================================
results = decode_predictions(
    predictions,
    top=5
)
# =========================================================
# DISPLAY RESULTS
# =========================================================
print("\nHasil Prediksi:")
for rank, (_, class_name, score) in enumerate(results[0], start=1):
    print(
        f"{rank}. {class_name}: "
        f"{score * 100:.2f}%"
    )