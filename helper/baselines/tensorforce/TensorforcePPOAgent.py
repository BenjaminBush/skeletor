from tensorforce.agents import PPOAgent
from tensorforce.core.networks import Network, Layer
from helper.wrappers import ClientToEnv, DictToListFull, JSONable

import tensorflow as tf
# from keras.models import Model
# from keras.layers import Conv1D, BatchNormalization, Activation, MaxPooling1D, Flatten, Dropout, Dense

from ...templates import TensorforceAgent

class Reshape(Layer):
    def __init__(self, target_shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def tf_apply(self, x, update):
        return tf.keras.layers.Reshape(self.target_shape)(x)


class Conv1D(Layer):
    def __init__(self, filters, kernel_size, strides, **kwargs):
        super(Conv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def tf_apply(self, x, update):
        x = tf.reshape(x, [1, 350])
        return tf.keras.layers.Conv1D(self.filters, self.kernel_size, self.strides)(x)

class RELU(Layer):
    def __init__(self, **kwargs):
        super(RELU, self).__init__(**kwargs)

    def tf_apply(self, x, update):
        return tf.keras.layers.Activation('relu')(x)

class CNN(Layer):
    def __init__(self, env_dim, **kwargs):
        super(CNN, self).__init__(**kwargs)
        self.env_dim = env_dim

    def tf_apply(self, x, update):
        # import pdb
        # pdb.set_trace()
        x = tf.keras.layers.Reshape((self.env_dim[0], 1))(x)
        x = tf.keras.layers.Conv1D(filters=32, kernel_size=(8), strides=2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Conv1D(filters=16, kernel_size=(8), strides=2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Conv1D(filters=8, kernel_size=(8), strides=1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Reshape((x.shape[1].value, -1))(x)
        x = tf.keras.layers.Flatten()(x)
        return x

class MaxPooling1D(Layer):
    def __init__(self, size, **kwargs):
        super(MaxPooling1D, self).__init__(**kwargs)
        self.pool_size = size

    def tf_apply(self, x, update):
        return tf.keras.layers.MaxPooling1D(self.pool_size)(x)


class TensorforcePPOAgent(TensorforceAgent):
    def __init__(self, observation_space, action_space,
                 directory='./TensorforcePPOAgent/'):
        # Create a Proximal Policy Optimization agent
        dense_network = []
        dense_network.append(dict(type='dense', size=256))
        dense_network.append(dict(type='dense', size=256))

        cnn_network = []
        cnn_network.append(dict(type='helper.baselines.tensorforce.TensorforcePPOAgent.Reshape', target_shape=(observation_space.shape[0], 1)))
        cnn_network.append(dict(type='conv1d', size=32, window=8, stride=2, activation='relu'))
        cnn_network.append(dict(type='helper.baselines.tensorforce.TensorforcePPOAgent.MaxPooling1D', size=2))
        cnn_network.append(dict(type='conv1d', size=16, window=8, stride=2, activation='relu'))
        cnn_network.append(dict(type='helper.baselines.tensorforce.TensorforcePPOAgent.MaxPooling1D', size=2))
        cnn_network.append(dict(type='conv1d', size=8, window=8, stride=1, activation='relu'))
        cnn_network.append(dict(type='flatten'))
        cnn_network.append(dict(type='dense', size=256))

        rnn_network = []
        rnn_network.append(dict(type='internal_lstm', size=256, dropout=0.3))
        rnn_network.append(dict(type='dense', size=256))





        self.agent = PPOAgent(
            states=dict(type='float', shape=observation_space.shape),
            actions=dict(type='float', shape=action_space.shape,
                         min_value=0, max_value=1),
            network=rnn_network,
            batching_capacity=1000,
            step_optimizer=dict(
                type='adam',
                learning_rate=1e-3
            )
        )
        self.directory = directory
