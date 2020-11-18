from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import numpy as np

def vae_loss(x, x_decoded_mean, z_log_var, z_mean, original_dim=784):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    k1_loss = -0.4 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + k1_loss

def sampling(args):
    z_mean, z_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0,)
    return z_mean + K.exp(z_var / 2) * epsilon

def get_decoder():
    """
    Decoder to convert from latent space into new image
    """
    input_node = Input(shape=(latent_dim,), name="input_decoder")
    decoder_h = Dense(intermediate_dimensions, activation='relu', name='decoder_h')(input_node)

    decoder_node = Dense(img_dimensions, activation='sigmoid', name="deocder_node")(decoder_h)
    decoder = Model(input_node, decoder_node, name='deocder_model')
    return decoder


def get_encoder():
    """
    Encoder an mnist image and define its latent space (z) mean and variance
    Returns:
        Encoder that defines a latent space (z) mean and variance

    """
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

"""
Define VAE model
"""
input_node = Input(shape=(img_dimensions,), name="input_encoder")
encoder = get_encoder()
decoder = get_decoder()
output_combined = decoder(encoder(input_node)[2])
vae = Model(input_node, output_combined)
vae.summary()
vae.compile(optimizer='rmsprop', loss=vae_loss)

