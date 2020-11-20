from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
tf.compat.v1.disable_eager_execution()


def vae_loss(x, x_decoded_mean):
    xent_loss = 784 * objectives.binary_crossentropy(x, x_decoded_mean)
    k1_loss = -0.5 * K.sum(1 + z_var - K.square(z_mean) - K.exp(z_var), axis=1)
    return xent_loss + k1_loss

def sampling(args):
    z_mean, z_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0,)
    return z_mean + K.exp(z_var / 2) * epsilon

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
intermediate_node = Dense(intermediate_dimensions, activation='relu', name="encoding")(input_node)
z_mean = Dense(latent_dim, name="mean")(intermediate_node)
z_var = Dense(latent_dim, name="variance")(intermediate_node)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_var])
encoder = Model(input_node, [z_mean, z_var, z], name="encoder")


input_node_2 = Input(shape=(latent_dim,), name="input_decoder")
decoder_h = Dense(intermediate_dimensions, activation='relu', name='decoder_h')(input_node_2)
decoder_node = Dense(img_dimensions, activation='sigmoid', name="deocder_node")(decoder_h)
decoder = Model(input_node_2, decoder_node, name='deocder_model')



output_combined = decoder(encoder(input_node)[2])
vae = Model(input_node, output_combined)
vae.summary()
vae.compile(optimizer='rmsprop', loss=vae_loss)

"""
Load MNIST Data
"""
print("-----\nLoading mnist data...")
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data = train_data.astype('float32')/255
test_data = test_data.astype('float32')/255

train_data = train_data.reshape( (len(train_data), np.prod(train_data.shape[1:])))
test_data = test_data.reshape( (len(test_data), np.prod(test_data.shape[1:])))


"""
Train VAE
"""
vae.fit(train_data, train_data, shuffle=True, epochs=10, batch_size=100, validation_data=(test_data, test_data), verbose=1) 

"""
display
"""
n = 15
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        img_decoded = decoder.predict(z_sample)
        digit = img_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size, j*digit_size: (j + 1) * digit_size] = digit
plt.figure(figsize=(10,10))
plt.imshow(figure, cmap='Greys_r')
plt.show()


