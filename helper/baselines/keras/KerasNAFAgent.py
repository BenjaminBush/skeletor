from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam, RMSprop
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from ...templates import KerasAgent


class KerasNAFAgent(KerasAgent):
    """
    An NAF agent using Keras library with Keras RL.
    """
    def __init__(self, observation_space, action_space, filename='KerasNAFAgent.h5f'):
        nb_actions = action_space.shape[0]
        
        # Set up the NNs        
        V_model = Sequential()
        V_model.add(Flatten(input_shape=(1,) + observation_space.shape))
        V_model.add(Dense(16))
        V_model.add(Activation('relu'))
        V_model.add(Dense(nb_actions))
        V_model.add(Activation('linear'))
        print("--------------V_model-------------")
        print(V_model.summary())
 
        mu_model = Sequential()
        mu_model.add(Flatten(input_shape=(1,) + observation_space.shape))
        mu_model.add(Dense(16))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(nb_actions))
        mu_model.add(Activation('linear'))
        print("--------------mu_model-------------")
        print(mu_model.summary())
     
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = concatenate([action_input, flattened_observation])
        x = Dense(16)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
        L_model = Model(input=[action_input, observation_input], output=x)
        print("--------------L_model-------------")
        print(L_model.summary())


        # Setup Keras RL's NAFAgent
        memory = SequentialMemory(limit=1000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
        self.agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                               memory=memory, nb_steps_warmup=50, random_process=random_process,
                               gamma=.99, target_model_update=1e-3)   
        self.agent.compile(Adam(lr=.001))

        self.filename = filename

        print(self.agent)
        #self.agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

        self.filename = filename
