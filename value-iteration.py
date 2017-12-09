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


def policy_to_action(state, policy):
    return policy[state]


def run_episode(env, policy, **kwargs):
    T = kwargs.get('T', 1000)
    gamma = kwargs.get('gamma', 0.99)
    render = kwargs.get('render', False)
    state = env.reset()
    total_rewards = 0

    for t in range(T):
        if render:
            env.render()
        action = policy_to_action(state, policy)
        state, reward, done, _ = env.step(action)
        total_rewards += pow(gamma, t) * reward
        if done:
            break
    return total_rewards


def eval_policy(env, policy, **kwargs):
    episode = kwargs.get('episode', 100)
    scores = [run_episode(env, policy, T=episode, **kwargs)
              for _ in range(episode)]
    return np.mean(scores)


def value_function(env, n_states, n_actions, **kwargs):
    # Keyword arguments
    eps = kwargs.get('eps', 1e-20)
    gamma = kwargs.get('gamma', 0.99)
    max_iter = kwargs.get('max_iter', 10000)
    # The Value says how good is it for an agent to be in a
    # particular state.
    V = np.zeros(shape=[n_states])
    for t in range(max_iter):
        # Copy over the Value at the previous time step
        prev_V = np.copy(V)
        # Calculate the value for each state
        for s in range(n_states):
            # Estimate how good it is to take an action in the current state
            V_sa = np.zeros(n_actions)  # value for taking n different actions
            for a in range(n_actions):
                # transition = [probability, next_state, reward, done]
                for transition in env.env.P[s][a]:
                    p, s_, r, _ = transition
                    # Expected future reward: ∑(E[r + µ*Q[s']])
                    V_sa[a] += p * (r + gamma * prev_V[s_])
            # Value of this state is the one that:
            # maximizes the value of the action taken in this state.
            V[s] = np.max(V_sa)
        # Convergence: The difference between the previous value and current value
        # is infinitesimally small i.e. difference is less than 1e-20, then break!
        if np.sum(np.fabs(prev_V - V)) <= eps:
            sys.stdout.write(f'\rValue iteration converged at {t+1:,} iteration')
            sys.stdout.flush()
            break
    return V


def extract_policy(env, value, n_states, n_actions, **kwargs):
    gamma = kwargs.get('gamma', 0.99)
    policy = np.zeros(shape=[n_states])
    for s in range(n_states):
        V_sa = np.zeros(shape=[n_actions])
        for a in range(n_actions):
            # transition = [probability, new_state, reward, done]
            for transition in env.env.P[s][a]:
                p, s_, r, _ = transition
                # Expected future reward: ∑(E[r + µ*Q[s']])
                V_sa[a] += p * (r + gamma * value[s_])
        # Best policy for this state is the index of the one that:
        # maximizes the value of being in this state and taken various actions.
        policy[s] = np.argmax(V_sa)
    return policy


if __name__ == '__main__':
    # Open AI's FrozenLake environment
    env_name = 'FrozenLake8x8-v0'
    env = gym.make(env_name)

    # Hyperparameters
    episodes = 1000
    n_actions = env.action_space.n
    n_states = env.observation_space.n
    print(f'{env_name} has {n_states} states & {n_actions} actions.')

    # Model
    optimal_value = value_function(env, n_states, n_actions)
    policy = extract_policy(env, optimal_value, n_states, n_actions)
    scores = eval_policy(env, policy, episode=episodes)

    # Logging
    print(f'\nAverage after {episodes:,} games = {scores:.2f}')
