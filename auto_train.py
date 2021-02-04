"""
This is the code to train the autoencoder with ReLU function
authors: Jianxin Wang

"""

from __future__ import absolute_import, division, print_function
import sys
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops



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


def create_model(input_size, hidden_size, code_size):
    inputs = tf.keras.Input(shape=(input_size,))
    #Encoder
    h_1 = tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu)(inputs)
    h = tf.keras.layers.Dense(code_size, activation=tf.nn.relu)(h_1)
    #Decoder
    h_2 = tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu)(h)
    output = tf.keras.layers.Dense(input_size, activation=tf.nn.sigmoid)(h_2)
    #output = tf_suplu(tf.keras.layers.Dense(input_size)(h_2))
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

checkpoint_path = "training_autoe/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
autoencoder = create_model(784, 128, 64)
autoencoder.fit(training_features, training_features, batch_size=128, epochs=5, callbacks = [cp_callback])
