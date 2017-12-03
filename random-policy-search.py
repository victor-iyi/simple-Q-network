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
    """
        Generate random policy.

        :return: 2-D policies [4x1]
         1st dimension is the `weights` while the
         2nd dimension is the `bias`
        """
    return [np.random.uniform(low=-1, high=1, size=4),
            np.random.uniform(low=-1, high=1)]


def policy_to_action(policy, obs):
    """
        Maps observation to action.

        :param obs: list
            Observation/state received by the agent.
        :param policy: list
            Policy to be followed by the agent.
        :return: action
         0 or 1. 0 - left, 1 - right.
        """
    value = np.dot(policy[0], obs) + policy[1]
    return 1 if value > 0 else 0


def run_episode(env, policy, T=5000, render=False):
    """
        Run episode for 5k timesteps.

        :param env: object
            OpenAI initialized environment object
        :param policy: list
            2-D list of values. [n_states x 1]
            Policy to be followed by agent.
        :param T: int default 50000
            Horizon or timesteps.
        :param render: boolean default False
            Display the environment as the agent performs
            actions. It slows down computation.
        :return: total_reward
            The accumulated total reward received by the agent.
        """
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
