"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  
  Created on 06 December, 2017 @ 10:35 AM.
  
  Copyright © 2017. Victor. All rights reserved.
"""
import gym
import numpy as np


def optimal_action(state, policy):
    return int(policy[state])


def eval_policy(env, policy, gamma=0.5, episodes=100):
    scores = [run_episode(env, policy, gamma, T=episodes)
              for _ in range(episodes)]
    return np.mean(scores)


def run_episode(env, policy, gamma=0.5, T=1000, render=False):
    total_rewards = 0
    state = env.reset()

    for t in range(T):
        if render:
            env.render()
        action = optimal_action(state, policy)
        state, reward, done, _ = env.step(action)
        total_rewards += (pow(gamma, t) * reward)
        if done:
            break
    return total_rewards


if __name__ == '__main__':
    env_name = 'FrozenLake8x8-v0'
    env = gym.make(env_name)
