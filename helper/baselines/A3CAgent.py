import tensorflow as tf
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import to_categorical
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import threading
import sys
import time
from ..templates import Agent



class Actor(object):
    def __init__(self, inp_dim, out_dim, network):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.model = self.build_actor_network(network)

        self.lr = 0.0001
        self.epsilon = 0.1
        self.rho = 0.99

        self.rms_optimizer = RMSprop(lr=self.lr, epsilon=self.epsilon, rho=self.rho)
        self.action_pl = K.placeholder(shape=(None, self.out_dim))
        self.advantages_pl = K.placeholder(shape=(None,))
        # Pre-compile before threading
        self.model._make_predict_function()

    def build_actor_network(self, network):
        """
        Assemble the Actor network to predict probability of each action
        """
        x = Dense(128, activation='relu')(network.output)
        out = Dense(self.outdim, activation='softmax')(x)
        return Model(network.input, out)

    def optimizer(self):
        weighted_actions = K.sum(self.action_pl*self.model.output, axis=1)
        eligibility = K.log(weighted_actions + 1e-10)*K.stop_gradient(self.advantages_pl)
        entropy = K.sum(self.model.output * K.log(self.model.output + 1e-10), axis=1)
        loss = 0.001*entropy - K.sum(eligibility)
        updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], loss)
        return K.function([self.model.input, self.action_pl, self.advantages_pl], [], updates=updates)

class Critic(object):
    def __init__(self, inp_dim, out_dim, network):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.model = self.build_critic_network(network)

        self.lr = 0.0001
        self.epsilon = 0.1
        self.rho = 0.99
        self.rms_optimizer = RMSprop(lr=self.lr, epsilon=self.epsilon, rho=self.rho)

        self.discounted_r = K.placeholder(shape=(None,))
        # Pre-compile before threading
        self.model._make_predict_function()

    def build_critic_network(self, network):
        """
        Assemble the Actor network to predict probability of each action
        """
        x = Dense(128, activation='relu')(network.output)
        out = Dense(1, activation='linear')(x)
        return Model(network.input, out)

    def optimizer(self):
        """
        MSE over discounted rewards
        """
        critic_loss = K.mean(K.square(self.discounted_r - self.model.output))
        updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], critic_loss)
        return K.function([self.model.input, self.discounted_r], [], updates=updates)

class A3CAgent(Agent):
    def __init__(self, observation_space, action_space, gamma=0.99):
        super(A3CAgent, self).__init__(observation_space, action_space)
        self.act_dim = action_space
        self.env_dim = observation_space
        self.gamma = gamma

        # Create the Actor and Critic Networks
        self.shared = self.build_network()
        self.actor = Actor(self.env_dim, self.act_dim, self.shared)
        self.critic = Critic(self.env_dim, self.act_dim, self.shared)

        # Build optimizers
        self.actor_opt = self.actor.optimizer()
        self.critic_opt = self.critic.optimizer()

        self.num_threads = 4
        self.training_interval = 50

    def build_network(self):
        """
        Assemble shared layers
        """
        inp = Input((self.env_dim))
        x = Flatten()(inp)
        x = Dense(64, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        return Model(inp, x)

    def act(self, observation):
        """
        Return action given an observation from the environment.
        """
        return np.random.choice(np.arrange(self.act_dim), 1, p=self.actor.predict(observation).ravel())

    def train_models(self, observations, actions, rewards, done):
        discounted_rewards = self.discount(rewards, done, observations[-1])
        state_values = self.critic.predict(np.array(observations))
        advtanges = discounted_rewards - np.reshape(state_values, len(state_values))
        self.actor_opt([observations, actions, advtanges])
        self.critic_opt([observations, discounted_rewards])

    def training_thread(self, agent, nb_steps, env, action_dim, training_interval, tqdm_e):
        global episode
        while episode < nb_steps:
            time, cumulative_reward, done = 0, 0, False
            old_state = env.reset()
            actions, states, rewards = [], [], []

            while not done:
                action = agent.act(np.expand_dims(old_state, axis=0))
                new_state, reward, done, _ = env.step(action)
                actions.append(to_categorical(action, action_dim))
                states.append(old_state)
                rewards.append(reward)

                old_state = new_state
                cumulative_reward += reward
                time += 1

                if (time % training_interval== 0 or done):
                    agent.train_models(states, actions, rewards, done)
                    actions, states, rewards = [], [], []


    def train(self, env, nb_steps):
        """
        Train agent for nb_steps.
        """

        # Instantiate one environment per thread
        envs = [deepcopy(env) for i in range(self.num_threads)]

        # Progress bar
        factor = 100.0/nb_steps
        tqdm_e = tqdm(range(nb_steps), desc='Score', leave=True, units=" episodes")

        threads = [threading.Thread(
            target=self.training_thread,
            args=(self,
                  nb_steps,
                  envs[i],
                  self.act_dim,
                  self.training_interval,
                  tqdm_e,
                  factor)) for i in range(self.num_threads)]

        for t in threads:
            t.start()
            time.sleep(1)
        [t.join() for t in threads]

        return None

    def discount(self, reward, done, observation):
        """
        Compute the gamma-discounted rewards over an episode
        """
        discounted_reward, cumulative_reward = np.zeros_like(reward), 0
        for t in reversed(range(0, len(reward))):
            cumulative_reward = reward[t] + cumulative_reward*self.gamma
            discounted_reward[t] = cumulative_reward
        return discounted_reward