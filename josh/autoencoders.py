from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras import Sequential, losses
import numpy as np
import cv2
import os
import sklearn.model_selection
import matplotlib.pyplot as plt

def load_images(folder, gray_scale = False):
    '''
    Function retreives images from folder returns them as numpy array
    Parameters:
        directory - string
        gray_scale - Bool
    Returns:
        np array of images
        '''
    images = []
    names = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            if gray_scale:
               img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append([img])
    return np.vstack(images)

def norm_split_images(images):
    '''
    Takes images and splits into training and validation
    Parameters:
        images - numpy array
    Returns:
        x_train, x_val - tuple of nump arrays
    '''
    images = images/255
    x_train, x_val = sklearn.model_selection.train_test_split(images,test_size=0.1, random_state=81)
    return x_train, x_val

def plot_images(image, color):
    plt.imshow(image, cmap = color)
    plt.axis("off")

def show_reconstructions(autoencoder, encoder, images, n_images = 5, color = None):
    '''
    Function prints original images in the validation set and their reconstruction
    Parameters:
        autoencoder - model
        number of images to be compared - int
        color - string (set to 'gray' for grayscale)
    '''
    reconstructions = autoencoder.predict(images[:n_images])
    code = encoder.predict(images[:n_images])
    code_shape = code[0].shape[0]
    fig = plt.figure(figsize = (n_images * 1.5, 4.5))
    for image_index in range(n_images):
        plt.subplot(3,n_images, 1 + image_index)
        plot_images(images[image_index], color)
        plt.subplot(3,n_images,1+n_images + image_index)
        plot_images(code[image_index].reshape((2,int(code_shape/2))), color)
        plt.subplot(3,n_images,1+2*n_images + image_index)
        plot_images(reconstructions[image_index], color)

def build_autoencoder(input_shape, latent_dim = 32):
    '''
    Function builds autoencoder. Encoder and Decoder have one hidden layer with 100 nodes.
    Parameters:
        input_shape - tuple of three
        latent_dim - input
    Returns:
        encoder,
        decoder,
        autoencoder
        '''
    encoder = Sequential([
        Flatten(input_shape = input_shape, name = "flatten"),
        Dense(500, activation = "selu", name = "dense1"),
        Dense(latent_dim, activation = "selu", name = "dense2")
    ], name = "encoder")

    decoder = Sequential([
        Dense(500,activation = "selu", input_shape = (latent_dim,), name = "dense3"),
        Dense(np.prod(input_shape),activation = "selu", name = "dense4"),
        Reshape(input_shape, name = "reshape")
    ], name = "decoder")

    autoencoder = Sequential([
        encoder,
        decoder
    ], name = "autoencoder")

    return encoder, decoder, autoencoder
