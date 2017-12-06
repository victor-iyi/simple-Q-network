"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  
  Created on 06 December, 2017 @ 10:35 AM.
  
  Copyright © 2017. Victor. All rights reserved.
"""
import sys

import gym
import numpy as np


def optimal_policy(state, policy):
    """
    Returns the action to take given a policy
    :param state: int
        Current state of the agent.
    :param policy: array
        The policy to pick actions from.
    :return: action int
        The optimal action given a state and policy

    """
    return int(policy[state])


def eval_policy(env, policy, **kwargs):
    """
    Evaluate how good a policy is

    :param env: object
        Initialized OpenAI's gym environment.
    :param policy: array
        The policy to be followed by the agent
    :param kwargs:
        :gamma float default 0.5
            Discount factor
        :episodes int default 100
            Number of episodes to play.
    :return:
    """
    # keyword arguments
    gamma = kwargs.get('gamma', default=0.5)
    episodes = kwargs.get('episodes', default=100)
    scores = [run_episode(env, policy, gamma, T=episodes)
              for _ in range(episodes)]
    return np.mean(scores)


def run_episode(env, policy, gamma=0.5, T=1000, render=False):
    """
    Run episodes for `T` timesteps

    :param env: object
        Initialized OpenAI's gym environment
    :param policy: array
        `n_states` dimensional array. The policy the agent
        should follow
    :param gamma:float
        Discount factor.
    :param T: int default 10k
        Times steps to run episodes.
    :param render: boolean default False
        Turn rendering on/off
    :return: rewards
        The total accumulated discounted reward
    """
    rewards = 0
    state = env.reset()

    for t in range(T):
        if render:
            env.render()
        action = optimal_policy(state, policy)
        state, reward, done, _ = env.step(action)
        rewards += (pow(gamma, t) * reward)
        if done:
            break
    return rewards


def value_iteration(env, n_states, n_actions, **kwargs):
    """
    Value iteration algorithm

    :param env: object
        Initialized OpenAI gym environment
    :param n_states: int
        Number of states
    :param n_actions:
        Number of actions
    :param kwargs:
        :gamma - float
            Discount factor
        :eps - float
            Epsilon for Epsilon Greedy exploration
        :max_iter - int default 10k
            Maximum number of iteration
    :return: V n_sates dimensional array
     Optimal Value of being in a state
    """
    # Keyword arguments
    gamma = kwargs.get('gamma', default=0.5)
    eps = kwargs.get('eps', default=1e-20)
    max_iter = kwargs.get('max_iter', default=10000)
    # Values of being in a state
    V = np.zeros(n_states)
    for t in range(max_iter):
        prev_v = np.copy(V)
        for s in range(n_states):
            # env.P[s][a] = transition function of being in a state
            # and taking an action.
            # it returns (probability, new_state, reward, info)
            Q_sa = [sum([p * (r + pow(gamma, t) * prev_v[s_])
                         for p, s_, r, _ in env.env.P[s][a]])
                    for a in range(n_actions)]
            V[s] = np.max(Q_sa)
        if np.sum(np.fabs(prev_v - V)) <= eps:
            sys.stdout.write('\rValue iteration converged at {i+1:}')
            sys.stdout.flush()
    return V


if __name__ == '__main__':
    env_name = 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    # Hyperparameters
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    gamma = 0.5
    # Get the optimal value function
    optimal_values = value_iteration(env, n_states, n_actions)
