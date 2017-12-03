"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  
  Created on 03 December, 2017 @ 8:43 PM.
  
  Copyright © 2017. Victor. All rights reserved.
"""
import gym
import numpy as np


def gen_policy():
    return np.random.choice(4, size=[16])


def policy_to_action(obs, policy):
    return policy[obs]


def run_episode(env, policy, T=1000, render=False):
    obs = env.render()
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
        total_rewards += run_episode(env, policy)  # , T=episodes)
    return total_rewards / episodes


def crossover(first, second):
    """
    Cross over

    :param first:
        First parent (dad)
    :param second:
        Second parent (mom)
    :return: offspring
        Crossed over offspring
    """
    offspring = np.copy(first)
    for i in range(len(first)):
        if np.random.uniform() > 0.5:
            offspring[i] = second[i]
    return offspring


def mutation(offspring, p=0.05):
    """
    Mutation.

    :param offspring:
        Offspring to be mutated
    :param p:
        Probability of mutation
    :return: mutated
        Maybe mutated offspring
    """
    mutated = np.copy(offspring)
    for i in range(len(offspring)):
        if np.random.choice(4) < p:
            mutated[i] = np.random.choice(4)
    return mutated

if __name__ == '__main__':
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)
    # seed random numbers
    np.random.seed(0)
    env.seed(0)
    # Hyperparameters
    n_policies = 500
    n_generations = 20
    n_fittest = 5
    # initial random policy
    policies = [gen_policy() for _ in range(n_policies)]
    # generation
    for gen in range(n_generations):
        # evaluate current generation
        scores = [eval_policy(env, p) for p in policies]
        print(f'Generation {gen+1:,} Max score = {max(scores):.2f}')
        # Select n_fittest best policies
        rank_idx = list(reversed(np.argsort(scores)))
        fittest = [policies[idx] for idx in rank_idx[:n_fittest]]
        selection_prob = np.array(scores) / np.sum(scores)
        # Perform crossover (returns n_policy - n_fittest) crossed-over offsprings
        offsprings = [crossover(policies[np.random.choice(range(n_policies), p=selection_prob)],
                                policies[np.random.choice(range(n_policies), p=selection_prob)])
                      for _ in range(n_policies - n_fittest)]
