from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate, Conv1D, MaxPooling1D, Dropout, Reshape
from keras.optimizers import Adam, RMSprop

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


from ...templates import KerasAgent


class KerasDDPGCNNAgent(KerasAgent):
    """
    An DDPG agent using Keras library with Keras RL.

    For more details about Deep Deterministic Policy Gradient algorithm, check
    "Continuous control with deep reinforcement learning" by Lillicrap.
    https://arxiv.org/abs/1509.02971
    """
    def __init__(self, observation_space, action_space, filename='KerasDDPGCNNAgent.h5f'):
        nb_actions = action_space.shape[0]

        # Actor network
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + observation_space.shape))
        actor.add(Dense(1024))
        actor.add(Activation('relu'))
        actor.add(Reshape((32,32)))
        actor.add(Conv1D(filters=32, kernel_size=4, activation='relu'))
        actor.add(Dropout(0.1))
        actor.add(MaxPooling1D(pool_size=2))
        actor.add(Flatten())
        actor.add(Dense(1024))
        actor.add(Activation('relu'))
        actor.add(Dense(nb_actions))
        actor.add(Activation('relu'))
        print(actor.summary())

        # Critic network
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = concatenate([action_input, flattened_observation])
        x = Dense(1024)(x)
        x = Reshape((32,32))(x)
        x = Conv1D(filters=32, kernel_size=4, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Activation('relu')(x)
        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('relu')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        print(critic.summary())

        # Setup Keras RL's DDPGAgent
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2,
                                                  size=nb_actions)
        self.agent = DDPGAgent(nb_actions=nb_actions,
                          actor=actor,
                          critic=critic,
                          critic_action_input=action_input,
                          memory=memory,
                          nb_steps_warmup_critic=100,
                          nb_steps_warmup_actor=100,
                          random_process=random_process,
                          gamma=.99,
                          target_model_update=1e-3,
                          delta_clip=1.)
        self.agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

        self.filename = filename