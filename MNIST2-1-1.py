# Example 4.9 - Experiment A
# Change filters from 32 to 64

import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape
X_train = X_train.reshape(
    (X_train.shape[0], 28, 28, 1)
).astype('float32')

X_test = X_test.reshape(
    (X_test.shape[0], 28, 28, 1)
).astype('float32')

# Normalize
X_train = X_train / 255
X_test = X_test / 255

# One hot encode
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]

# Create model
model = Sequential()

# 64 filters instead of 32
model.add(Conv2D(
    64,
    (5, 5),
    input_shape=(28, 28, 1),
    activation='relu'
))

model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Summary
model.summary()

# Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Training
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200
)

# Evaluation
scores = model.evaluate(X_test, y_test, verbose=0)

print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
print("CNN Accuracy: %.2f%%" % (scores[1] * 100))