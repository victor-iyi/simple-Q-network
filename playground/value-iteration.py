"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  
  Created on 07 December, 2017 @ 4:33 PM.
  
  Copyright © 2017. Victor. All rights reserved.
"""
import gym
import numpy as np


def run_episode(env, policy, **kwargs):
    # Keyword arguments
    T = kwargs.get('T', 10000)
    gamma = kwargs.get('gamma', 0.99)
    render = kwargs.get('render', False)

    total_rewards = 0
    state = env.reset()
    for t in range(T):
        if render:
            env.render()
        action = int(policy[state])
        state, reward, done, _ = env.step(action)
        total_rewards += pow(gamma, t) * reward
        if done:
            break
    return total_rewards


def eval_policy(env, policy, **kwargs):
    episodes = kwargs.get('episodes', 100)
    scores = [run_episode(env, policy, **kwargs) for _ in range(episodes)]
    return np.mean(scores)


def value_function(env, n_states, n_actions, **kwargs):
    eps = kwargs.get('eps', 1e-20)
    gamma = kwargs.get('gamma', 0.99)
    max_iter = kwargs.get('max_iter', 1000)
    V = np.zeros(shape=[n_states])
    for t in range(max_iter):
        v = np.copy(V)
        for s in range(n_states):
            V_sa = np.zeros(n_actions)
            for a in range(n_actions):
                # transition = [probability, state, reward, info]
                for transition in env.env.P[s][a]:
                    p, s_, r, _ = transition
                    V_sa[a] += p * (r + gamma * v[s_])
            V[s] = np.max(V_sa)
        # Convergence
        if np.sum(np.fabs(V - v)) <= eps:
            print(f'Solution found at {t+1:,} iteration')
            break
    return V


if __name__ == '__main__':
    env_name = 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    # Hyperparameters
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    # The Model
    optimal_policy = value_function(env, n_states, n_actions)
    score = eval_policy(env, optimal_policy)
