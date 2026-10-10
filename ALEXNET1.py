# Example 4.5 AlexNet - Modified Dense Layer Units

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

# Fully Connected Layers - MODIFIED
model.add(Flatten())

# 1st Fully Connected Layer
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(17, activation='softmax'))

# Display model summary
model.summary()

# Compile the model
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=["accuracy"]
)