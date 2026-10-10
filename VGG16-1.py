# Example 4.10 - VGG19

import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# Load VGG19 model with ImageNet weights
model = VGG19(weights='imagenet')

# Display model summary
model.summary()