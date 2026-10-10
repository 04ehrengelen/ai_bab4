
# Example 4.20 - SimpleRNN3.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# Mengubah data menjadi dataset matrix
def convertToMatrix(data, step):
    X, Y = [], []

    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d])
        Y.append(data[d])

    return np.array(X), np.array(Y)


# Membaca dataset Air Passengers
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, usecols=[1])

# Menampilkan data awal
print(df.head())

# Visualisasi data asli
plt.figure(figsize=(10, 5))
plt.plot(df.values)
plt.title("Monthly Air Passengers")
plt.xlabel("Month")
plt.ylabel("Number of Passengers")
plt.show()


# Menentukan jumlah time steps
step = 4

N = df.shape[0]
Tp = int(N * 0.8)

values = df.values

# Membagi data menjadi training dan testing
train = values[0:Tp, :]
test = values[Tp:N, :]

# Menambahkan elemen untuk membantu pembentukan sequence
test = np.append(test, np.repeat(test[-1, :], step))
train = np.append(train, np.repeat(train[-1, :], step))

# Mengubah data menjadi input dan target
trainX, trainY = convertToMatrix(train, step)
testX, testY = convertToMatrix(test, step)

# Mengubah bentuk input menjadi (samples, timesteps, features)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print("Train X shape:", trainX.shape)
print("Train Y shape:", trainY.shape)
print("Test X shape:", testX.shape)
print("Test Y shape:", testY.shape)


# Membuat model SimpleRNN
model = Sequential()

model.add(
    SimpleRNN(
        units=32,
        input_shape=(1, step),
        activation="relu"
    )
)

model.add(Dense(8, activation="relu"))
model.add(Dense(1))

# Compile model
model.compile(
    loss="mean_squared_error",
    optimizer="rmsprop"
)

# Menampilkan ringkasan model
model.summary()

# Melatih model
model.fit(
    trainX,
    trainY,
    epochs=10,
    batch_size=16,
    verbose=2
)

# Melakukan prediksi
trainPredict = model.predict(trainX, verbose=0)
testPredict = model.predict(testX, verbose=0)

# Menggabungkan hasil prediksi
predicted = np.concatenate((trainPredict, testPredict), axis=0)

print("Prediction shape:", predicted.shape)

# Menghitung nilai loss pada data training
trainScore = model.evaluate(trainX, trainY, verbose=0)
print("Training Loss:", trainScore)

# Membandingkan data asli dengan hasil prediksi
plt.figure(figsize=(12, 6))

plt.plot(df.values, label="Actual Data")
plt.plot(predicted, label="Prediction")
plt.axvline(x=Tp, color="red", linestyle="--", label="Train/Test Split")

plt.title("SimpleRNN - Air Passengers Prediction")
plt.xlabel("Month")
plt.ylabel("Number of Passengers")
plt.legend()
plt.tight_layout()
plt.show()
