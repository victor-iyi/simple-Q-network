"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  
  Created on 03 December, 2017 @ 7:47 PM.
  
  Copyright © 2017. Victor. All rights reserved.
"""
import gym
import numpy as np


def gen_random_policy():
    return [np.random.uniform(low=-1, high=1, size=4),
            np.random.uniform(low=-1, high=1)]


def policy_to_action(policy, obs):
    value = np.dot(policy[0], obs) + policy[1]
    return 1 if value > 0 else 0


def run_episode(env, policy, T=5000, render=False):
    obs = env.reset()
    total_reward = 0
    for t in range(T):
        if render:
            env.render()
        action = policy_to_action(policy, obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

if __name__ == '__main__':
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    # Generate random policies
    n_policy = 500
    policies = [gen_random_policy() for _ in range(n_policy)]
    # Score of each policy
    scores = [run_episode(env, p) for p in policies]
    # Best score/ policy
    best_score = max(scores)
    best_policy = policies[scores.index(best_score)]

    print(f'Best Score = {best_score}')
    print(f'Running with best policy...')
    run_episode(env, best_policy, render=True)
