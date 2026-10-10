# Example 4.16 - AutoencoderKeras.py
# MNIST Image Reconstruction using Autoencoder

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input
from tensorflow.keras.models import Sequential, Model


# ==========================================
# 1. Load MNIST Dataset
# ==========================================

(x_train, _), (x_test, _) = mnist.load_data()

# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# ==========================================
# 2. Build Autoencoder
# ==========================================

def build_autoencoder(img_shape, code_size):

    # Encoder: compress the image
    encoder = Sequential([
        Input(shape=img_shape),
        Flatten(),
        Dense(int(code_size), activation="relu")
    ], name="encoder")

    # Decoder: reconstruct the image
    decoder = Sequential([
        Input(shape=(int(code_size),)),
        Dense(int(np.prod(img_shape)), activation="sigmoid"),
        Reshape(img_shape)
    ], name="decoder")

    return encoder, decoder


# ==========================================
# 3. Create the Model
# ==========================================

IMG_SHAPE = x_train[0].shape
CODE_SIZE = 32

encoder, decoder = build_autoencoder(
    IMG_SHAPE,
    CODE_SIZE
)

# Connect encoder and decoder
inp = Input(shape=IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(
    inputs=inp,
    outputs=reconstruction,
    name="autoencoder"
)

# Compile the model
autoencoder.compile(
    optimizer="adamax",
    loss="mse"
)

print("\nAutoencoder Model Summary:")
autoencoder.summary()


# ==========================================
# 4. Train the Autoencoder
# ==========================================

print("\nStarting model training...")

history = autoencoder.fit(
    x=x_train,
    y=x_train,
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

print("\nTraining completed!")


# ==========================================
# 5. Display Training and Validation Loss
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Autoencoder Model Loss")
plt.ylabel("Mean Squared Error (MSE)")
plt.xlabel("Epoch")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================
# 6. Display Original and Reconstructed Images
# ==========================================

def show_image(img):
    plt.imshow(
        np.clip(img, 0, 1),
        cmap="gray",
        vmin=0,
        vmax=1
    )
    plt.axis("off")


def visualize(img, encoder, decoder):

    # Encode the original image
    code = encoder.predict(
        img[np.newaxis, ...],
        verbose=0
    )[0]

    # Reconstruct the image
    reco = decoder.predict(
        code[np.newaxis, ...],
        verbose=0
    )[0]

    plt.figure(figsize=(10, 3))

    # Original image
    plt.subplot(1, 3, 1)
    plt.title("Original")
    show_image(img)

    # Encoded representation
    plt.subplot(1, 3, 2)
    plt.title("Encoded Representation")
    plt.imshow(
        code.reshape(4, 8),
        cmap="viridis",
        aspect="auto"
    )
    plt.axis("off")

    # Reconstructed image
    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    show_image(reco)

    plt.tight_layout()
    plt.show()


# Display the first five test images
for i in range(5):
    print(f"Displaying test image {i + 1}")

    img = x_test[i]
    visualize(img, encoder, decoder)