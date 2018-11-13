from .Agent import Agent
import math
from sklearn.linear_model import LinearRegression
import pandas as pd
from pandas import DataFrame, read_csv

class TensorforceAgent(Agent):
    def __init__(self, observation_space, action_space, directory):
        """
        Template class for agents using Keras RL library.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.directory = directory
        self.agent = None
        self.shared_acts = [4,5,6,9,10,13,14,16]
        self.frame = pd.read_csv('Experimental walking.csv', sep=';')
        act_frame = self.frame.drop(['Pelvic Tilt', 'Pelvic Up/Down Obl',
            'Pelvic Int/Ext Rot', 'Hip Flex/Ext', 'Hip Flex/Ext (L)', 'Hip Ad/Ab',
            'Hip Ad/Ab (L)', 'Hip Int/Ext Rot', 'Hip Int/Ext Rot (L)',
            'Knee Flex/Ext','Knee Flex/Ext (L)', 'Ankle Dorsi/Plant',
            'Ankle Dorsi/Plant (L)'],axis=1)

        obs_frame = self.frame.drop(['rect_fem_r', 'rect_fem_l', 'hamstrings_r',
        'hamstrings_l', 'bifemsh_r', 'bifemsh_l', 'tib_ant_l', 'gastroc_l'],
        axis=1)
        self.linreg = LinearRegression()
        self.linreg.fit(obs_frame.get_values(), act_frame.get_values())
        

    def train(self, env, nb_steps):
        try:
            print('[train] Loading weights from {}'.format(self.directory))
            self.agent.restore_model(directory=self.directory)
            print('[train] Successfully loaded weights from {}'.format(self.directory))
        except ValueError:
            print('[train] Pretrained model {} not found. Starting from scratch.'.format(self.directory))

        print('[train] Training \'{}\''.format(type(self).__name__))
        step_count = 0
        episode_count = 1
        while step_count < nb_steps:
            episode_step_count = 0
            obs = env.reset()
            done = False
            total_rew = 0
            while not done:
                action = self.agent.act(obs)
                obs, rew, done, info = env.step(action)
                predicted_action = predict_action(obs)
                mod_rew += exp_comp(action, predicted_action, scaling = 100.0 - (step_count * 0.00002)                  total_rew += rew
                if step_count < 5000000:
                    self.agent.observe(reward=mod_rew, terminal=done)
                else:
                    self.agent.observe(reward=rew, terminal=done)
                episode_step_count += 1
            step_count += episode_step_count
            print('[train] Episode {:3} | Steps Taken: {:3} | Total Steps: Taken {:6}/{:6} | Total reward: {}'.format(
                episode_count, episode_step_count, step_count, nb_steps, total_rew))
            episode_count += 1
        print('[train] Finished training')

        print('[train] Saved weights to \'{}\''.format(self.directory))
        self.agent.save_model(directory=self.directory)
        print('[train] Successfully saved weights to \'{}\''.format(self.directory))

    def test(self, env):
        """
        Run agent locally.
        """
        try:
            print('[test] Loading weights from {}'.format(self.directory))
            self.agent.restore_model(directory=self.directory)
            print('[test] Successfully loaded weights from {}'.format(self.directory))
        except ValueError:
            print('[test] Unable to find pretrained model {}. Aborting.'.format(self.directory))
            return

        print('[test] Running \'{}\''.format(type(self).__name__))
        obs = env.reset()
        done = False
        total_rew = 0
        while not done:
            action = self.agent.act(obs)
            obs, rew, done, info = env.step(action)
            total_rew += rew
            self.agent.observe(reward=rew, terminal=done)
        print('[test] Total reward: ' + str(total_rew))
        print('[test] Finished test.')

        print('[test] Saved weights to \'{}\''.format(self.directory))
        self.agent.save_model(directory=self.directory)
        print('[test] Successfully saved weights to \'{}\''.format(self.directory))

    def submit(self, env):
        """
        Submit agent to CrowdAI server.
        """
        try:
            print('[submit] Loading weights from \'{}\''.format(self.directory))
            self.agent.restore_model(directory=self.directory)
            print('[submit] Successfully loaded weights from \'{}\''.format(self.directory))
        except ValueError:
            print('[submit] Unable to find pretrained model from \'{}\'. Aborting.'.format(self.directory))
            return

        print('[submit] Running \'{}\''.format(type(self).__name__))
        obs = env.reset()
        episode_count = 1
        step_count = 0
        total_rew = 0
        try:
            while True:
                action = self.act(obs)
                obs, rew, done, info = env.step(action)
                total_rew += rew
                step_count += 1
                if done:
                    print('[submit] Episode {} | Steps Taken: {:3} | Total reward: {}'.format(episode_count, step_count, total_rew))
                    obs = env.reset()
                    episode_count += 1
                    step_count = 0
                    total_rew = 0
        except TypeError:
            # When observation is None - no more steps left
            pass

        print('[submit] Finished running \'{}\' on Server environment. Submitting results to server...'.format(type(self).__name__))
        env.submit()
        print('[submit] Submitted results successfully!')

    def act(self, obs):
        return self.agent.act(obs)

    def shrink_act(vec): #hard-coded to remove actions we can't predict 
        return [vec[i] for i in self.shared_acts]

    def exp_comp(act_vec, exp_act_vec, scaling=1):
        rmse = math.sqrt(sum([(act_vec[i] - exp_act_vec[i])**2]) / len(self.shared_acts))
        return ((1 - rmse) * scaling)

    def predict_action(obs):
        a = obs['joint_pos']
        red_obs = [a['ground_pelvis'][5], a['ground_pelvis'][3],
            a['ground_pelvis'][4], a['hip_r'][0], a['hip_l'][0], a['hip_r'][1],
            a['hip_l'][1], a['hip_r'][2], a['hip_l'][2], a['knee_r'], a['knee_l'],
            a['ankle_r'], a['ankle_l']]

        return(self.linreg.predict(red_obs)