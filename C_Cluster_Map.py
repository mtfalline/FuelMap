import cv2
import numpy as np
import tarfile
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.mixture import GaussianMixture

IMAGES_NAME = "ProsserCove/ProsserCoveTest4.tar"
IMAGES_TEST = "ProsserCove/ProsserCoveTest3.tar"
c1 = 0

encoder = load_model('encoders/c5e_encoder_256n5.h5')
decoder = load_model('decoders/c5e_decoder_256n5.h5')


def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_map_dataset(IMAGES):
    all_photos = []

    with tarfile.open(IMAGES) as f:
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
    plt.imshow(code.reshape([code.shape[-1]//4, -1]))

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    show_image(reco)
    global c1
    if c1 < 0:
        plt.show()
        c1 += 1


icodes = []
newX = load_map_dataset(IMAGES_NAME)
newX = newX.astype('float32') / 255.0 - 0.5

xCodes = []
for aNewX in newX:
    xCodes.append(encoder.predict(aNewX[None])[0])

print('length of xCodes', len(xCodes))

X = load_map_dataset(IMAGES_TEST)
X = X.astype('float32') / 255.0 - 0.5
print(X.max(), X.min())

for i in range(40):
    img = X[i]
    visualize(img, encoder, decoder)

nxcodes = []

for index, axcode in enumerate(xCodes):
    nxcodes.append(axcode.reshape((6*6*32)))

print(nxcodes[0].shape)

print(icodes[0].shape)

ncodes = []

for index, acode in enumerate(icodes):
    ncodes.append(acode.reshape((6*6*32)))

print(ncodes[0].shape)

gmm = GaussianMixture(n_components=8, max_iter=10).fit(nxcodes)
labels = gmm.predict(ncodes)

probs = gmm.predict_proba(ncodes)
print(probs[:55].round(3))

brush = [0, 0, 0, 0, 0]
brushSparsePine = [0, 0, 0, 0, 0]
densePine = [0, 0, 0, 0, 0]
dryGrass =[0, 0, 0, 0, 0]
pine = [0, 0, 0, 0, 0]
sand = [0, 0, 0, 0, 0]
sparseBrush = [0, 0, 0, 0, 0]
water = [0, 0, 0, 0, 0]

for i in range(5):
    for j in range(5):
        if labels[i] == labels[j]:
            brush[i] += 1
        if labels[i+5] == labels[j+5]:
            brushSparsePine[i] += 1
        if labels[i+10] == labels[j+10]:
            densePine[i] += 1
        if labels[i+15] == labels[j+15]:
            dryGrass[i] += 1
        if labels[i+20] == labels[j+20]:
            pine[i] += 1
        if labels[i+25] == labels[j+25]:
            sand[i] += 1
        if labels[i+30] == labels[j+30]:
            sparseBrush[i] += 1
        if labels[i+35] == labels[j+35]:
            water[i] += 1

print('brush: ', max(brush), brush)
print('brushSparsePine: ', max(brushSparsePine), brushSparsePine)
print('densePine: ', max(densePine), densePine)
print('dryGrass: ', max(dryGrass), dryGrass)
print('pine: ', max(pine), pine)
print('sand: ', max(sand), sand)
print('sparseBrush: ', max(sparseBrush), sparseBrush)
print('water: ', max(water), water)

score1 = (max(brush) + max(brushSparsePine) + max(densePine) + max(dryGrass) +
          max(pine) + max(sand) + max(sparseBrush) + max(water))

print('score1+: ', score1)

print(labels)

cluster = np.empty(8, dtype=int)

cluster[0] = labels[brush.index(max(brush))]
cluster[1] = labels[brushSparsePine.index(max(brushSparsePine))+5]
cluster[2] = labels[densePine.index(max(densePine))+10]
cluster[3] = labels[dryGrass.index(max(dryGrass))+15]
cluster[4] = labels[pine.index(max(pine))+20]
cluster[5] = labels[sand.index(max(sand))+25]
cluster[6] = labels[sparseBrush.index(max(sparseBrush))+30]
cluster[7] = labels[water.index(max(water))+35]

clusterSize = np.empty(8, dtype=int)

print(cluster)
clusterMatch = 0
for i in range(8):
    for j in range(i+1, 8):
        if cluster[i] == cluster[j]:
            clusterMatch += 1

print('score2-:', clusterMatch)




