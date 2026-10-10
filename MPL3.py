# Example 4.3
# Multiple Layer Perceptron Classification
# Modified: Dense layer uses 5 neurons instead of 10

import time

from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load dataset
whole_data = load_breast_cancer()

X_data = whole_data.data
y_data = whole_data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_data,
    y_data,
    test_size=0.3,
    random_state=7
)

# Create MLP model
model = Sequential()

# Changed from 10 neurons to 5 neurons
model.add(Dense(
    5,
    activation='relu',
    input_shape=[X_train.shape[1]]
))

model.add(Dense(5, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Start measuring training time
start_time = time.time()

# Train model
model.fit(
    X_train,
    y_train,
    batch_size=50,
    validation_split=0.2,
    epochs=100,
    verbose=1
)

# Calculate training time
training_time = time.time() - start_time

# Evaluate model
results = model.evaluate(X_test, y_test, verbose=0)

print("Training time:", training_time, "seconds")
print("Loss:", results[0])
print("Accuracy:", results[1])