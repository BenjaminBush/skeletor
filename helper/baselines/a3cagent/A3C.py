import sys
import gym
import math
import time
import threading
import numpy as np
import pandas as pd

from tqdm import tqdm
from keras.models import Model, load_model
from keras import regularizers
from keras.layers import Input, Dense, Flatten, Reshape, Conv1D, Conv2D, LSTM, MaxPooling1D, Dropout, BatchNormalization, Activation
from keras import backend

from sklearn.linear_model import LinearRegression

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
        #linear model
        self.frame = pd.read_csv('Experimental walking.csv', sep=';')
        act_frame = self.frame.drop([
             'Hip Flex/Ext', 'Hip Flex/Ext (L)', 'Hip Ad/Ab',
            'Hip Ad/Ab (L)', 'Hip Int/Ext Rot', 'Hip Int/Ext Rot (L)',
            'Knee Flex/Ext', 'Knee Flex/Ext (L)', 'Ankle Dorsi/Plant',
            'Ankle Dorsi/Plant (L)'], axis=1)
        obs_frame = self.frame.drop(['rect_fem_r', 'rect_fem_l', 'hamstrings_r',
            'hamstrings_l', 'bifemsh_r', 'bifemsh_l', 'tib_ant_l', 'gastroc_l'],
            axis=1)
        self.linreg = LinearRegression()
        self.linreg.fit(obs_frame.values, act_frame.values)


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

            # DENSE
            # x = inp
            # x = Reshape((self.env_dim[0], -1))(x)
            # x = Dense(2048, activation='relu')(x)
            # x = Reshape((backend.int_shape(x)[1], -1))(x)
            # x = Flatten()(x)

            # CNN
            # x = inp
            # x = Reshape((self.env_dim[0], -1))(x)
            # x = Conv1D(filters=32, kernel_size=8, strides=2)(x)
            # x = BatchNormalization()(x)
            # x = Activation("relu")(x)
            # x = MaxPooling1D(pool_size=2)(x)
            # x = Conv1D(filters=16, kernel_size=8, strides=2)(x)
            # x = BatchNormalization()(x)
            # x = Activation("relu")(x)
            # x = MaxPooling1D(pool_size=2)(x)
            # x = BatchNormalization()(x)
            # x = Conv1D(filters=8, kernel_size=8, strides=1, padding="valid")(x)
            # x = BatchNormalization()(x)
            # x = Activation("relu")(x)
            # x = Flatten()(x)
            # x = Dropout(0.4)(x)
            # x = Dense(256, activation='relu')(x)
            # x = Reshape((backend.int_shape(x)[1], -1))(x)
            # x = Flatten()(x)

            # RNN
            x = inp
            x = Reshape((self.env_dim[0], -1))(x)
            x = LSTM(128, activation='relu', return_sequences=True)(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.4)(x)
            x = Dense(64, activation='relu')(x)
            x = Reshape((backend.int_shape(x)[1], -1))(x)
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

    def act(self, observation):
        observation = np.asarray(observation)
        preds = self.actor.predict(observation)
        preds = np.asarray(preds)
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
        n_threads = 4
        nb_episodes = 500
        training_interval = 32
        consecutive_frames = 0


        envs = [ProstheticsEnv(visualize=False) for i in range(n_threads)]
        [e.reset() for e in envs]

        state_dim = envs[0].get_observation_space_size()
        action_dim = envs[0].get_action_space_size()

        # Create threads
        factor = 100.0 / (nb_episodes)
        tqdm_e = tqdm(range(nb_episodes), desc='Score', leave=True, unit=" episodes")

        threads = [threading.Thread(target=training_thread, args=
            (self, nb_episodes, envs[i], action_dim, training_interval,
            summary_writer, tqdm_e, factor, 25.0)) for i in range(n_threads)]

        for t in threads:
            t.start()
            time.sleep(1)
        [t.join() for t in threads]

        return self.actor, self.critic

    def pred_reshape(self, x):
        inp_dim = self.env_dim
        if len(x.shape) < 4 and len(inp_dim) > 2: return np.expand_dims(x, axis=0)
        elif len(x.shape) < 2: return np.expand_dims(x, axis=0)
        else: return x

    def test(self, env):
        """
        Run agent locally.
        """
        print('[test] Running \'{}\''.format(type(self).__name__))
        observation = env.reset()

        model = load_model('A3CAgent_actor.h5')

        total_reward = 0
        done = False
        while not done:
            observation = np.asarray(observation)
            observation = self.pred_reshape(observation)
            action = model.predict(observation)
            action = np.asarray(action)[0]
            # action = model.predict(observation)[0]
            observation, reward, done, info = env.step(action)
            total_reward += reward

        print('[test] Total Reward of \'{}\': {}'.format(type(self).__name__,
                                                        total_reward))

    def get_linreg_rmse(self, obs, action):
        #keep only the muscles, joints we can predict
        pred_actions = [4,5,6,9,10,13,14,16]
        small_action = [[action[i] for i in pred_actions]] #need column vector
        small_obs = [obs[i] for i in [80,71,81,72,82,73,92,89,65,62]]
        for i in range(len(small_obs)):
            small_obs[i] = small_obs[i] * (3.14159 / 180)
            if i in [2,3]:
                small_obs[i] = small_obs[i] * -1
        exp_action = self.linreg.predict([small_obs]) #gets column vector
        return np.sqrt(sum([(small_action[0][i] - exp_action[0][i]) ** 2 for i in
            range(len(pred_actions)) ]) / len(pred_actions))
