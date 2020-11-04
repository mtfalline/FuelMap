import cv2
import numpy as np
import tarfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Reshape, Input
from keras.models import Sequential, Model

IMAGE_TAR_NAME = "images2.tar"


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
    encoder.add(Flatten(input_shape=img_shape))
    encoder.add(Dense(code_size))

    decoder = Sequential()
    decoder.add(Dense(np.prod(img_shape), input_shape=(code_size,)))
    decoder.add(Reshape(img_shape))

    return encoder, decoder


X = load_map_dataset()
X = X.astype('float32') / 255.0 - 0.5
print(X.max(), X.min())
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
IMG_SHAPE = X.shape[1:]

encoder, decoder = build_autoencoder(IMG_SHAPE, 32)

inp = Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp, reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')

print(autoencoder.summary())

history = autoencoder.fit(x=X_train, y=X_train, epochs=20, validation_data=[X_test, X_test])

encoder.save('encoder1.h5')
decoder.save('decoder1.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




