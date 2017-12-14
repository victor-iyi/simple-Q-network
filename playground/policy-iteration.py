"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  
  Created on 14 December, 2017 @ 2:43 PM.
  
  Copyright © 2017. Victor. All rights reserved.
"""
import gym
import numpy as np


def run_episode(env, policy, **kwargs):
    """
    Run a single episode for some time steps
    :param env: object
        Initialized OpenAI's gym environment.
    :param policy: ndarray
        Policy to be followed by the agent.
    :param kwargs:
     :T int default 1k
        Time steps which this episode lasts
     :render boolean default False
        Display the game play or not
     :discount float default None
        (gamma) discount factor
    :return: total_reward:
        Total reward for this episode
    """
    # kwargs
    T = kwargs.get('T', 1000)
    render = kwargs.get('render', False)
    discount = kwargs.get('discount', None)

    total_reward = 0
    state = env.reset()
    for t in range(T):
        if render:
            env.render()
        action = policy[state]
        state, reward, done, _ = env.step(action)
        total_reward += pow(discount, t) * reward if discount else reward
        if done:
            break
    return total_reward


def eval_policy(env, policy, episodes=100):
    scores = [run_episode(env, policy, T=episodes)
              for _ in range(episodes)]
    return np.mean(scores)


if __name__ == '__main__':
    # Environment
    env_name = 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    # Hyperparameters
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    print(f'{env_name} has {n_states} states and {n_actions} possible actions')
