#!/usr/bin/env python3
"""
You can run RandomAgent locally with following command:

    ./run.py RandomAgent

You can run FixedActionAgent with visuals with following command:

    ./run.py FixedActionAgent -v

You can submit FixedActionAgent with visuals with following command:

    ./run.py FixedActionAgent -s
"""
import argparse

from osim.env import ProstheticsEnv
from osim.http.client import Client

from helper.wrappers import ClientToEnv, DictToListFull, ForceDictObservation, JSONable
from helper.CONFIG import remote_base, crowdai_token
from helper.baselines import *
from agents import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from helper.baselines.utils.networks import get_session
import os
from keras.models import load_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run or submit agent.')
    parser.add_argument('agent', help='specify agent\'s class name.')
    parser.add_argument('-t', '--train', action='store', dest='nb_steps',
                        help='train agent locally')
    parser.add_argument('-s', '--submit', action='store_true', default=False,
                        help='submit agent to crowdAI server')
    parser.add_argument('-v', '--visualize', action='store_true', default=False,
                        help='render the environment locally')
    parser.add_argument('-test', '--test', action='store', dest='test', help='Test the agent locally')
    args = parser.parse_args()

    if args.agent not in globals():
        print(globals())
        raise ValueError('[run] Agent {} not found.'.format(args.agent))

    if args.agent != 'A3C':
        SpecifiedAgent = globals()[args.agent]

        if args.submit and args.nb_steps:
            raise ValueError('[run] Cannot train and submit agent at same time.')

        if args.submit and args.visualize:
            raise ValueError('[run] Cannot visualize agent while submitting.')

        if args.submit:
            # Submit agent
            client = Client(remote_base)
            client.env_create(crowdai_token, env_id='ProstheticsEnv')
            client_env = ClientToEnv(client)
            client_env = DictToListFull(client_env)
            client_env = JSONable(client_env)
            agent = SpecifiedAgent(client_env.observation_space,
                                   client_env.action_space)
            agent.submit(client_env)
        elif args.nb_steps:
            # Train agent locally
            env = ProstheticsEnv(visualize=args.visualize)
            env = ForceDictObservation(env)
            env = DictToListFull(env)
            env = JSONable(env)
            agent = SpecifiedAgent(env.observation_space, env.action_space)
            agent.train(env, int(args.nb_steps))
        else:
            # Test agent locally
            env = ProstheticsEnv(visualize=args.visualize)
            env = ForceDictObservation(env)
            env = DictToListFull(env)
            env = JSONable(env)
            agent = SpecifiedAgent(env.observation_space, env.action_space)
            agent.test(env)
    else:
        if args.test:
            env = ProstheticsEnv(visualize=args.visualize)
            observation = env.reset()
            state_dim = env.get_observation_space_size()
            action_dim = env.get_action_space_size()
            agent = A3C(action_dim, state_dim, 0)
            agent.test(env)
        else:
            set_session(get_session())
            os.system('rm -rf A3C')
            summary_writer = tf.summary.FileWriter("A3C/tensorboard_" + "ProstheticsEnv")

            env = ProstheticsEnv(visualize=args.visualize)
            observation = env.reset()
            state_dim = env.get_observation_space_size()
            action_dim = env.get_action_space_size()
            agent = A3C(action_dim, state_dim, 0)
            actor, critic = agent.train(env, summary_writer)
            actor.model.save('A3CAgent_actor.h5')
            critic.model.save('A3CAgent_critic.h5')
