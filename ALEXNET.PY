# Example 4.5 AlexNet with Keras
# Modified from:
# https://engmrk.com/alexnet-implementation-using-keras/

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization

# Create the AlexNet model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(
    filters=96,
    input_shape=(224, 224, 3),
    kernel_size=(11, 11),
    activation='relu',
    strides=(4, 4),
    padding='valid'
))

model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='valid'
))

# 2nd Convolutional Layer
model.add(Conv2D(
    filters=256,
    kernel_size=(11, 11),
    activation='relu',
    strides=(1, 1),
    padding='valid'
))

model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='valid'
))

# 3rd Convolutional Layer
model.add(Conv2D(
    filters=384,
    kernel_size=(3, 3),
    activation='relu',
    strides=(1, 1),
    padding='valid'
))

# 4th Convolutional Layer
model.add(Conv2D(
    filters=384,
    kernel_size=(3, 3),
    activation='relu',
    strides=(1, 1),
    padding='valid'
))

# 5th Convolutional Layer
model.add(Conv2D(
    filters=256,
    kernel_size=(3, 3),
    activation='relu',
    strides=(1, 1),
    padding='valid'
))

model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='valid'
))

# 1st Fully Connected Layer
model.add(Flatten())

model.add(Dense(
    4096,
    activation='relu'
))

model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(
    4096,
    activation='relu'
))

model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(
    1000,
    activation='relu'
))

model.add(Dropout(0.4))

# Output Layer
model.add(Dense(
    17,
    activation='softmax'
))

# Display model summary
model.summary()

# Compile the model
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy']
)

# Fit the model
# model.fit()

# Prediction with the model
# model.evaluate()
