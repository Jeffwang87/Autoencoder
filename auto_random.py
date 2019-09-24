from __future__ import absolute_import, division, print_function
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
input_size = 784
hidden_size = 128
code_size = 64


def suplu(x):
    if x < 0:
        return 0
    elif x >= 0 and x <1:
        return x**2
    else:
        return x**3
np_suplu = np.vectorize(suplu)

def d_suplu(x):
    if x < 0:
        return 0
    elif x >= 0 and x <1:
        return 2*x
    else:
        return 3*x**2
np_d_suplu = np.vectorize(d_suplu)

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+2))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def suplu_grad(op, grad):
    x = op.inputs[0]
    n_gr = tf_d_suplu(x)    #defining the gradient.
    return grad * n_gr

np_suplu_32 = lambda x: np_suplu(x).astype(np.float32)
def tf_suplu(x,name=None):
    with tf.name_scope(name, "suplu", [x]) as name:
        y = py_func(np_suplu_32,   #forward pass function
                        [x],
                        [tf.float32],
                        name=name,
                         grad= suplu_grad) #the function that overrides gradient
        y[0].set_shape(x.get_shape())     #when using with the code, it is used to specify the rank of the input.
        return y[0]

np_d_suplu_32 = lambda x: np_d_suplu(x).astype(np.float32)
def tf_d_suplu(x,name=None):
    with tf.name_scope(name, "d_suplu", [x]) as name:
        y = tf.py_func(np_d_suplu_32,
                        [x],
                        [tf.float32],
                        name=name,
                        stateful=False)
        return y[0]


(training_features, _), _ = tf.keras.datasets.mnist.load_data()
training = training_features / np.max(training_features)
training_features = training[0:50000]
training_features = training_features.reshape(training_features.shape[0],
                                    training_features.shape[1]*training_features.shape[2]).astype(np.float32)
testing = training[50000:]

inputs = tf.keras.Input(shape=(input_size,))
# Ecoder
h_1 = tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu)(inputs)
h = tf.keras.layers.Dense(code_size, activation=tf.nn.relu)(h_1)

#Decoder
h_2 = tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu)(h)
#output = tf.keras.layers.Dense(input_size, activation=tf.nn.sigmoid)(h_2)
output = tf_suplu(tf.keras.layers.Dense(input_size)(h_2))

autoencoder = tf.keras.Model(inputs=inputs, outputs=output)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(training_features, training_features, batch_size=128, epochs=5)
origin = testing
testing = testing.reshape(testing.shape[0], testing.shape[1]*testing.shape[2]).astype(np.float32)
predicted = autoencoder.predict(testing)
plt.figure(figsize=(40, 4))
for i in range(10):
    # display original
    ax = plt.subplot(2, 20, i + 1)
    plt.imshow(origin[i])
    plt.gray()
    # display reconstructed
    ax = plt.subplot(2, 20, i + 1 + 20)
    plt.imshow(predicted[i].reshape(28, 28))
    plt.gray()
plt.show()
