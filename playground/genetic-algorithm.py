"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  
  Created on 04 December, 2017 @ 5:52 PM.
  
  Copyright © 2017. Victor. All rights reserved.
"""
import time

import gym
import numpy as np


def gen_policies(states, actions):
    return np.random.choice(actions, size=[states])


def policy_to_action(policy, state):
    return policy[state]


def run_episode(env, policy, T=1000, render=False):
    total_rewards = 0
    state = env.reset()
    for _ in range(T):
        if render:
            env.render()
        action = policy_to_action(policy, state)
        state, reward, done, _ = env.step(action)
        total_rewards += reward
        if done:
            break
    return total_rewards


def eval_policy(env, policy, episodes=100):
    total_rewards = 0
    for _ in range(episodes):
        total_rewards += run_episode(env, policy, T=episodes)
    return total_rewards / episodes


def crossover(first, second, p=0.5):
    offspring = np.copy(first)
    for i in range(len(offspring)):
        if np.random.uniform() < p:
            offspring[i] = second[i]
    return offspring


def mutate(offspring, action, p=0.05):
    mutant = np.copy(offspring)
    for i in range(len(mutant)):
        if np.random.choice(action) < p:
            mutant[i] = np.random.choice(action)
    return mutant


if __name__ == '__main__':
    env_name = 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    # Seed random numbers
    env.seed(0)
    np.random.seed(0)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    # Hyperparameters
    n_policies = 200
    n_generations = 10
    n_fittest = 10
    # Loop thought the population
    start = time.time()
    policies = [gen_policies(n_states, n_actions) for _ in range(n_policies)]
    for gen in range(n_generations):
        # Evaluate the population
        scores = [eval_policy(env, policy) for policy in policies]
        print(f'Generation {gen+1:,}\tScore = {max(scores):.2%}')
        # Select n best
        rank_idx = list(reversed(np.argsort(scores)))
        fittest = [policies[idx] for idx in rank_idx]
        # Crossover
