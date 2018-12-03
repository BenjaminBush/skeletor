""" Training thread for A3C
"""

import numpy as np

from keras.utils import to_categorical
from ..utils.networks import tfSummary

episode = 0

def training_thread(agent, Nmax, env, action_dim, f, summary_writer,
    tqdm, factor, scaling=0.0):
    """ Build threads to run shared computation across
    """

    global episode
    step_count = 0
    while episode < Nmax:
        # Reset episode
        time, cumul_reward, done = 0, 0, False
        old_state = env.reset()
        actions, states, rewards, mod_rewards = [], [], [], []
        #separate mod_rewards so we don't break whatever tqdm does
        while not done:
            # Actor picks an action (following the policy)
            action = agent.policy_action(np.expand_dims(old_state, axis=0))[0]
            # Fill NaNs with 0's. Not sure why we are getting nans but this avoids any training issues
            where_nans = np.isnan(action)
            action[where_nans] = 0
            # Retrieve new state, reward, and whether the state is terminal
            new_state, r, done, _ = env.step(action)
            step_count += 1
            # Memorize (s, a, r) for training
            actions.append(action)
            if scaling * step_count > 0.0:
            #CHANGE THIS BACK TO OLD)STATE
                mod_r = agent.get_linreg_rmse(old_state, action) * (scaling - step_count * 0.1)
            #to stop modification at 5,000, pass scaling=500
                mod_rewards.append(r + mod_r)
            rewards.append(r)
            states.append(old_state)
            # Update current state
            old_state = new_state
            cumul_reward += r
            time += 1
            # Asynchronous training
            if(time%f==0 or done):
                if scaling != 0.0:
                    agent.train_models(states, actions, mod_rewards,done)
                else:
                    agent.train_models(states, actions, rewards, done)
                actions, states, rewards, mod_rewards = [], [], [], []

        # Export results for Tensorboard
        score = tfSummary('score', cumul_reward)
        summary_writer.add_summary(score, global_step=episode)
        summary_writer.flush()
        #
        tqdm.set_description("Score: " + str(cumul_reward))
        tqdm.update(int(episode * factor))
        episode += 1
