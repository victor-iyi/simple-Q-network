"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  
  Created on 07 December, 2017 @ 4:33 PM.
  
  Copyright © 2017. Victor. All rights reserved.
"""
import gym
import numpy as np


def optimal_policy(state, policy):
    """
    Optimal policy to take in a given state.

    :param state: int
        Current state of the agent.
    :param policy:
        (optimal) policy to be followed by the agent.
    :return: action, int
        Action to be taken from the policy given a state.
    """
    return policy[state]


def run_episode(env, policy, **kwargs):
    """
    Play game for `T` horizon

    :param env: object
        Initialized OpenAI's gym environment
    :param policy: ndarray
        Policy to be used by the agent
    :param kwargs:
        :gamma: float, default 0.5
            Discount factor.
        :T: int, default 10k
            Horizon/timesteps
        render: boolean, default False
    :return:
    """
    # Keyword arguments
    gamma = kwargs.get('gamma', default=0.5)
    T = kwargs.get('T', default=10000)
    render = kwargs.get('render', default=False)
    total_rewards = 0
    state = env.reset()
    # Game loop.
    for t in range(T):
        if render:
            env.render()
        action = optimal_policy(state, policy)
        state, reward, done, _ = env.step(action)
        total_rewards += (pow(gamma, t) * reward)
        if done:
            break
    return total_rewards


def eval_policy(env, policy, **kwargs):
    """
    Evaluate how good a policy is.

    :param env: object
        Initialized OpenAI's gym environment.
    :param policy: ndarray
        Policy to be used by the agent.
    :param kwargs:
        :gamma: float, default 0.5
            Discount factor.
        :episodes: int, default 100
            Number of episodes to be run.
    :return: average score
    """
    gamma = kwargs.get('gamma', default=0.5)
    episodes = kwargs.get('episodes', default=100)
    scores = [run_episode(env, policy, gamma=gamma, T=episodes)
              for _ in range(episodes)]
    return np.mean(scores)


if __name__ == '__main__':
    # Environment
    env_name = 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    # Hyperparameters
