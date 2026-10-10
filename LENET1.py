# Example 4.4 LeNet-5 Keras
# Modified: Different number of filters
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D
model = Sequential()
# Convolutional Layer 1
# Original: filters=6
# Modified: filters=8
model.add(Conv2D(
    filters=8,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(32, 32, 1)
))

# Average Pooling Layer 1
model.add(AveragePooling2D(pool_size=(2, 2)))
# Convolutional Layer 2
# Original: filters=16
# Modified: filters=20
model.add(Conv2D(
    filters=20,
    kernel_size=(3, 3),
    activation='relu'
))

# Average Pooling Layer 2
model.add(AveragePooling2D(pool_size=(2, 2)))

# Flatten
model.add(Flatten())

# Fully Connected Layers
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# Display model summary
model.summary()