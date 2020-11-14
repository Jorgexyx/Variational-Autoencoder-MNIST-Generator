from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import numpy as np

def sampling(args):
    z_mean, z_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0,)
    return z_mean + K.exp(z_var / 2) * epsilon

def get_encoder(img_dimensions, intermediate_dimensions, latent_dim):
    """
    Encoder an mnist image and define its latent space (z) mean and variance
    Returns:
        Encoder that defines a latent space (z) mean and variance

    """
    input_node = Input(shape=(img_dimensions,), name="input")
    intermediate_node = Dense(intermediate_dimensions, activation='relu', name="encoding")(input_node)

    # define latent space
    z_mean = Dense(latent_dim, name="mean")(intermediate_node)
    z_var = Dense(latent_dim, name="variance")(intermediate_node)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_var])
    encoder = Model(input_node, [z_mean, z_var, z], name="encoder")
    return encoder

"""
Defiine hyperparameters
"""
img_dimensions = 784 # 28 * 28
intermediate_dimensions = 256
latent_dim = 2
batch_size = 100

get_encoder(img_dimensions, intermediate_dimensions, latent_dim)

