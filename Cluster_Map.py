import cv2
import numpy as np
import tarfile
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

IMAGES_NAME = "images2.tar"
c1 = 0

encoder = load_model('encoder1.h5')
decoder = load_model('decoder1.h5')


def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_map_dataset():
    all_photos = []

    with tarfile.open(IMAGES_NAME) as f:
        for m in f.getmembers():
            if m.isfile() and m.name.endswith(".jpg"):
                img = decode_image_from_raw_bytes(f.extractfile(m).read())
                all_photos.append(img)

    all_photos = np.stack(all_photos).astype('uint8')

    return all_photos


def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))


def visualize(img, encoder, decoder):
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    icodes.append(code)

    plt.subplot(1, 3, 1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1, 3, 2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//8, -1]))

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    show_image(reco)
    global c1
    if c1 < 10:
        plt.show()
        c1 += 1


icodes = []
X = load_map_dataset()
X = X.astype('float32') / 255.0 - 0.5
print(X.max(), X.min())

for i in range(250):
    img = X[i]
    visualize(img, encoder, decoder)

kmeans = KMeans(n_clusters=5, random_state=0).fit(icodes)
print(kmeans.labels_)

gmm = GaussianMixture(n_components=5).fit(icodes)
labels = gmm.predict(icodes)

probs = gmm.predict_proba(icodes)
print(probs[:20].round(3))


