from time import time
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate, Conv1D, MaxPooling1D, Dropout, Reshape, LSTM
from keras.optimizers import Adam, RMSprop
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from ...templates import KerasAgent


class KerasNAFCNNAgent(KerasAgent):
    """
    An NAF agent using Keras library with Keras RL.
    """
    def __init__(self, observation_space, action_space, filename='KerasNAFCNNAgent.h5f'):
        nb_actions = action_space.shape[0]
        

        lstm_out = 20
        V_model = Sequential()
        V_model.add(Flatten(input_shape=(1,) + observation_space.shape))
        V_model.add(Reshape((14,25)))
        V_model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        V_model.add(Dropout(0.1))
        V_model.add(MaxPooling1D(pool_size=2))
        V_model.add(Flatten())
        V_model.add(Dense(1024))
        V_model.add(Activation('relu'))
        V_model.add(Dense(1))
        V_model.add(Activation('relu'))


        

        mu_model = Sequential()
        mu_model.add(Flatten(input_shape=(1,) + observation_space.shape))
        mu_model.add(Reshape((14,25)))
        mu_model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        mu_model.add(Dropout(0.1))
        mu_model.add(MaxPooling1D(pool_size=2))
        mu_model.add(Flatten())
        mu_model.add(Dense(1024))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(nb_actions))
        mu_model.add(Activation('relu'))

        
        
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + observation_space.shape,
                              name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = concatenate([action_input, flattened_observation])
        x = Dense(1024)(x)
        x = Reshape((32,32))(x)
        x = Conv1D(filters=32, kernel_size=8, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Activation('sigmoid')(x)
        x = Dense(1024)(x)
        x = Activation('sigmoid')(x)
        x = Dense(1024)(x)
        x = Activation('sigmoid')(x)
        x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
        x = Activation('sigmoid', name='L_final')(x)
        L_model = Model(inputs=[action_input, observation_input], outputs=x)


        # Setup Keras RL's NAFAgent
        memory = SequentialMemory(limit=1000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
        self.agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                               memory=memory, nb_steps_warmup=32, random_process = random_process,
                               gamma=.99, target_model_update=1e-3)   
        self.agent.compile(Adam(lr=1e-3))


        self.callbacks = [TensorBoard(log_dir='./tmp/log')]

        #self.agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
        self.filename = filename