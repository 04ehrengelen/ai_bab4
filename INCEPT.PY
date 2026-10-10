# Example 4.6 - InceptionV3

from tensorflow.keras.applications import InceptionV3

# Init the model
model = InceptionV3(weights='imagenet')

# Display model summary
model.summary()