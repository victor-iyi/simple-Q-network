"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  
  Created on 04 December, 2017 @ 5:52 PM.
  
  Copyright © 2017. Victor. All rights reserved.
"""
import gym
import numpy as np


def gen_policies():
    return np.random.choice(4, size=[16])


def policy_to_action(obs, policy):
    return policy[obs]


def run_episode(env, policy, T=1000, render=False):
    obs = env.reset()
    total_rewards = 0
    for _ in range(T):
        if render:
            env.render()
        action = policy_to_action(obs, policy)
        obs, reward, done, _ = env.step(action)
        total_rewards += reward
        if done:
            break
    return total_rewards


def eval_policy(env, policy, episodes=100):
    total_rewards = 0
    for _ in range(episodes):
        total_rewards += run_episode(env, policy, T=episodes)
    return total_rewards / episodes


if __name__ == '__main__':
    env_name = 'MountainLake-v0'
    env = gym.make(env_name)
    # seeding random number
    np.random.seed(0)
    env.seed(0)
    # Hyperparameters
    n_policies = 500
    n_generations = 20
    n_fittest = 5
    # Generate a policy
    policies = [gen_policies() for _ in range(n_policies)]
    # Genetic Algorithm: Loop though each generations...
    for gen in range(n_generations):
        # 1. Evaluation: Evaluate population
        scores = [eval_policy(env, p) for p in policies]
        print(f'Generation {gen+1:,} Max score = {max(scores)}')
        # 2. Selection: Select the fittest policy
        rank_idx = list(reversed(np.argsort(scores)))
        fittest = [policies[idx] for idx in rank_idx[:n_fittest]]
        prob = np.array(scores) / np.sum(scores)
