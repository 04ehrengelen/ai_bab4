#Example 4.10

import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# Create VGG16 model
model = VGG16(weights='imagenet')

# Display model summary
model.summary()