from tensorforce.agents import PPOAgent
from helper.wrappers import ClientToEnv, DictToListFull, JSONable

from ...templates import TensorforceAgent, TensorforceAgent_exp


class TensorforcePPOAgent_exp(TensorforceAgent_exp):
    def __init__(self, observation_space, action_space,
                 directory='./TensorforcePPOAgent_exp/'):
        TensorforceAgent_exp.__init__(self, observation_space,
            action_space, directory='./TensorforcePPOAgent_exp/')
        # Create a Proximal Policy Optimization agent
        self.agent = PPOAgent(
            states=dict(type='float', shape=observation_space.shape),
            actions=dict(type='float', shape=action_space.shape,
                         min_value=0, max_value=1),
            network=[
                dict(type='dense', size=256),
                dict(type='dense', size=256),
            ],
            batching_capacity=1000,
            step_optimizer=dict(
                type='adam',
                learning_rate=1e-3
            )
        )
        self.directory = directory
