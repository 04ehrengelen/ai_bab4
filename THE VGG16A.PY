# Example 4.11

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# Load model VGG16
model = VGG16(weights='imagenet')

# Lokasi gambar
img_path = r'D:\KAMPUS\SEMESTER 5\COMPUTER VISION\AI\image\Optimus Prime.jpg'

# Load gambar
img = image.load_img(img_path, target_size=(224, 224))

# Ubah gambar menjadi array
x = image.img_to_array(img)

# Tambahkan dimensi batch
x = np.expand_dims(x, axis=0)

# Preprocessing sesuai VGG16
x = preprocess_input(x)

# Prediksi
predictions = model.predict(x)

# Ambil 5 prediksi teratas
results = decode_predictions(predictions, top=5)

# Tampilkan hasil
print("\nHasil Prediksi:")
for _, class_name, score in results[0]:
    print(f"{class_name}: {score * 100:.2f}%")