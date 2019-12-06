from keras.layers import Input, Dense
from keras.models import Model
from matplotlib import pyplot as plt
from keras.callbacks import TensorBoard
from keras import regularizers

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# # Add a hidden layer
# hidden_layer_en = Dense(512, activation='relu', input_shape=(784,),
#                 activity_regularizer=regularizers.l1(0.0000001))(input_img)
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

noise_factor = 0.2
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

tensorboard = TensorBoard(log_dir=f".\logs\Tensors", write_graph=True)

history = autoencoder.fit(x_train_noisy, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_noisy))
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test_noisy)
decoded_imgs = decoder.predict(encoded_imgs)

plt.figure(1)
plt.imshow(x_test[0].reshape((28, 28)), cmap='gray')
plt.title('Original Image :')
plt.show(block=False)

plt.figure(2)
plt.imshow(x_test_noisy[0].reshape((28, 28)), cmap='gray')
plt.title('Noisy Image :')
plt.show(block=False)


plt.figure(3)
plt.imshow(decoded_imgs[0].reshape((28, 28)), cmap='gray')
plt.title('Decoded Image :')
plt.show(block=False)

print(history.history.keys())

plt.figure(4)
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['loss'], label='Loss')
plt.legend(loc='upper left')
plt.show(block=False)
plt.figure(5)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.legend(loc='upper left')
plt.show()