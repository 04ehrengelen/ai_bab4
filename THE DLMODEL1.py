# Example 4.7 - Modified

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

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

model.add(MaxPooling2D())
model.add(Dropout(0.2))

# Flatten
model.add(Flatten())

# Fully Connected Layers
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Display model summary
model.summary()