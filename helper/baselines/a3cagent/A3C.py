import sys
import gym
import time
import threading
import numpy as np

from tqdm import tqdm
from keras.models import Model
from keras import regularizers
from keras.layers import Input, Dense, Flatten, Reshape, Conv1D, Conv2D, LSTM
from keras import backend

from .Critic import Critic
from .Actor import Actor
from .Thread import training_thread
# from utils.atari_environment import AtariEnvironment
# from utils.continuous_environments import Environment
from helper.baselines.utils.networks import conv_block
# from utils.stats import gather_stats
from osim.env import ProstheticsEnv


class A3C:
    """ Asynchronous Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, k, gamma = 0.99, lr = 0.0001):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = (k+env_dim,)
        self.gamma = gamma
        # Create actor and critic networks
        self.shared = self.buildNetwork()
        self.actor = Actor(self.env_dim, act_dim, self.shared, lr)
        self.critic = Critic(self.env_dim, act_dim, self.shared, lr)
        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()

    def buildNetwork(self):
        """ Assemble shared layers
        """
        inp = Input((self.env_dim))
        # If we have an image, apply convolutional layers
        if(len(self.env_dim) > 2):
            x = Reshape((self.env_dim[1], self.env_dim[2], -1))(inp)
            x = conv_block(x, 32, (2, 2))
            x = conv_block(x, 32, (2, 2))
            x = Flatten()(x)
        else:
            #x = Flatten()(inp)
            x = inp
            #x = Reshape((self.env_dim[1], self.env_dim[2], -1))(inp)
            # This should be a 3d convolution f000k but not really because it's returned as a list
            x = Reshape((self.env_dim[0], -1))(x)
            x = Conv1D(filters=16, kernel_size=8, strides=4, padding="valid", activation='elu')(x)
            x = Conv1D(filters=32, kernel_size=8, strides=2, padding="valid", activation='elu')(x)
            x = Flatten()(x)
            x = Dense(256, activation='elu')(x)
            x = Reshape((backend.int_shape(x)[1], -1))(x)
            x = LSTM(256, activation='elu', return_sequences=True)(x)
            x = Flatten()(x)
        return Model(inp, x)

    def policy_action(self, s):
        """ Use the actor's network to predict the next action to take, using the policy
        """
        preds = self.actor.predict(s)
        preds = np.asarray(preds)
        #print("preds are {}".format(preds))
        #return np.random.choice(np.arange(self.act_dim), 1, p=self.actor.predict(s)).ravel()
        return preds

    def discount(self, r, done, s):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train_models(self, states, actions, rewards, done):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(rewards, done, states[-1])
        state_values = self.critic.predict(np.array(states))
        advantages = discounted_rewards - np.reshape(state_values, len(state_values))
        # Networks optimization
        self.a_opt([states, actions, advantages])
        self.c_opt([states, discounted_rewards])

    def train(self, env, summary_writer):

        # Instantiate one environment per thread
        n_threads = 1
        nb_episodes = 10
        training_interval = 32
        consecutive_frames = 0


        envs = [ProstheticsEnv(visualize=False) for i in range(n_threads)]
        [e.reset() for e in envs]

        state_dim = envs[0].get_observation_space_size()
        action_dim = envs[0].get_action_space_size()

        # Create threads
        factor = 100.0 / (nb_episodes)
        tqdm_e = tqdm(range(nb_episodes), desc='Score', leave=True, unit=" episodes")

        threads = [threading.Thread(
                target=training_thread,
                args=(self,
                    nb_episodes,
                    envs[i],
                    action_dim,
                    training_interval,
                    summary_writer,
                    tqdm_e,
                    factor)) for i in range(n_threads)]

        for t in threads:
            t.start()
            time.sleep(1)
        [t.join() for t in threads]

        return self.actor, self.critic