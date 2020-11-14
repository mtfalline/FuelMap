import cv2
import numpy as np
import tarfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Reshape, Input, Conv2D, MaxPool2D, UpSampling2D
from keras.models import Sequential, Model

IMAGE_TAR_NAME = "ProsserCove/ProsserCovePatches.tar"
# IMAGE_TAR_NAME = "images2.tar"


def decode_image(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def load_map_dataset():
    all_photos = []

    with tarfile.open(IMAGE_TAR_NAME) as f:
        for m in f.getmembers():
            if m.isfile() and m.name.endswith(".jpg"):
                img = decode_image(f.extractfile(m).read())
                all_photos.append(img)

    all_photos = np.stack(all_photos).astype('uint8')

    return all_photos


def build_autoencoder(img_shape, code_size):

    encoder = Sequential()
    # encoder.add(Conv2D(1024, (3, 3), activation='relu', padding='same', input_shape=img_shape))
    # encoder.add(MaxPool2D((2, 2), padding='same'))
    encoder.add(Conv2D(512, (3, 3), activation='relu', padding='same', input_shape=img_shape))
    encoder.add(MaxPool2D((2, 2), padding='same'))
    encoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    encoder.add(MaxPool2D((2, 2), padding='same'))
    encoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    encoder.add(MaxPool2D((2, 2), padding='same'))
    encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    encoder.add(MaxPool2D((2, 2), padding='same'))
    encoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    encoder.add(MaxPool2D((2, 2), padding='same'))
    # encoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    # encoder.add(MaxPool2D((2, 2), padding='same'))

    decoder = Sequential()
    # decoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    # decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    decoder.add(UpSampling2D((2, 2)))
    # decoder.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    # decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(3, (3,3), activation='sigmoid', padding='same'))

    return encoder, decoder


X = load_map_dataset()
X = X.astype('float32') / 255.0 - 0.5
print(X.max(), X.min())
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
IMG_SHAPE = X.shape[1:]

encoder, decoder = build_autoencoder(IMG_SHAPE, 8)

inp = Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp, reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')

print(autoencoder.summary())

# Add noise
noise_factor = 0.8
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

# X_train_noisy = np.clip(X_train_noisy, -0.5, 0.5)
# X_test_noisy = np.clip(X_test_noisy, -0.5, 0.5)

history = autoencoder.fit(x=X_train_noisy, y=X_train, epochs=2, validation_data=[X_test_noisy, X_test])

# history = autoencoder.fit(x=X_train, y=X_train, epochs=2, validation_data=[X_test, X_test])

encoder.save('encoders/c2e_encoder_512n8.h5')
decoder.save('decoders/c2e_decoder_512n8.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




