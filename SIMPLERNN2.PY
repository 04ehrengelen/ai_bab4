
# Example 4.19 - SimpleRNN2.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Activation
from tensorflow.keras.optimizers import Adam

# Membuat model Sequential
model = Sequential()

# Layer SimpleRNN
model.add(
    SimpleRNN(
        50,
        input_shape=(49, 1),
        return_sequences=False
    )
)

# Layer Dense dengan 46 neuron
model.add(Dense(46))

# Fungsi aktivasi Softmax
model.add(Activation('softmax'))

# Optimizer Adam
adam = Adam(learning_rate=0.001)

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy']
)

# Menampilkan ringkasan model
model.summary()
