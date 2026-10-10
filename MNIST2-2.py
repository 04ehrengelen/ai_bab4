# Example 4.9 - Modified
# Add another Conv2D, MaxPooling2D, and Dropout layer

import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape to be [samples][width][height][channels]
X_train = X_train.reshape(
    (X_train.shape[0], 28, 28, 1)
).astype('float32')

X_test = X_test.reshape(
    (X_test.shape[0], 28, 28, 1)
).astype('float32')

# Normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# One hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]

# Create model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(
    32,
    (5, 5),
    input_shape=(28, 28, 1),
    activation='relu'
))

model.add(MaxPooling2D())
model.add(Dropout(0.2))

# 2nd Convolutional Layer - NEW
model.add(Conv2D(
    64,
    (3, 3),
    activation='relu'
))

# 2nd MaxPooling Layer - NEW
model.add(MaxPooling2D())

# 2nd Dropout Layer - NEW
model.add(Dropout(0.2))

# Flatten
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(128, activation='relu'))

# Output Layer
model.add(Dense(num_classes, activation='softmax'))

# Display model summary
model.summary()

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Fit the model
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200
)

# Evaluation of the model
scores = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
print("CNN Accuracy: %.2f%%" % (scores[1] * 100))