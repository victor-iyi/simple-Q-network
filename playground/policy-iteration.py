"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  
  Created on 14 December, 2017 @ 2:43 PM.
  
  Copyright © 2017. Victor. All rights reserved.
"""
import sys

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
    """
    Evaluate the current policy

    :param env: object
        Initialized OpenAI's gym environment.
    :param policy: ndarray
        Policy to be followed by the Agent
    :param episodes: int
        Number of episodes to evaluate
    :return:
        mean accumulated reward
    """
    scores = [run_episode(env, policy, T=episodes)
              for _ in range(episodes)]
    return np.mean(scores)


def extract_value(env, policy, n_states, **kwargs):
    """
    Extract the utility/value of a policy

    :param env: object
        Initialized OpenAI's gym environment
    :param policy: ndarray
        Policy to be extract it's utility
    :param n_states: int
        Total states as provided by the environment
    :param kwargs:
        :eps: float default 1e-10
            For epsilon greedy exploration
        :gamma: float default 0.99
            (MDP) discount factor
        :max_iter: int default 1k
            Maximum iteration
    :return: V
        Value of the given policy
    """
    # keyword arguments
    eps = kwargs.get('eps', 1e-10)
    gamma = kwargs.get('gamma', 0.99)
    max_iter = kwargs.get('max_iter', 1000)
    V = np.zeros(shape=[n_states])
    for t in range(max_iter):
        v = np.copy(V)
        # go through all states
        for s in range(n_states):
            a = policy[s]
            for trans in env.env.P[s][a]:
                p, s_, r, _ = trans
                V[s] += p * (r + gamma * v[s_])
        # convergence
        if np.sum(np.fabs(v - V)) <= eps:
            sys.stdout.write(f'\rValue extraction converged @ {t+1} iter\n')
            sys.stdout.flush()
            break
    return V


def extract_policy(env, V, n_states, n_actions, **kwargs):
    gamma = kwargs.get('gamma', 0.99)
    policy = np.zeros(shape=[n_states])
    for s in range(n_states):
        V_sa = np.zeros(shape=[n_actions])
        for a in range(n_actions):
            for trans in env.env.P[s][a]:
                p, s_, r, _ = trans
                V_sa[a] += p * (r + gamma * V[s_])
        policy[s] = np.argmax(V_sa)
    return policy


def policy_iteration(env, n_states, n_actions, **kwargs):
    max_iter = kwargs.get('max_iter', 1000)
    policy = np.random.choice(n_actions, size=[n_states])
    for t in range(max_iter):
        old_policy = extract_value(env, policy, n_states)
        new_policy = extract_policy(env, old_policy, n_states, n_actions)
        if np.all(old_policy == new_policy):
            sys.stdout.write(f'\rPolicy iteration converged @ {t+1}')
            sys.stdout.flush()
            break

    return policy


if __name__ == '__main__':
    # Environment
    env_name = 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    # Hyperparameters
    episodes = 100
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    print(f'{env_name} has {n_states} states and {n_actions} possible actions')
    # Policy iteration
    policy = policy_iteration(env, n_states, n_actions)
    score = eval_policy(env, policy, episodes=episodes)
    print(f'After {episodes:,} episodes, acc = {score:.2f}')
