"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  
  Created on 03 December, 2017 @ 8:43 PM.
  
  Copyright © 2017. Victor. All rights reserved.
"""
import time

import gym
import numpy as np


def gen_policy():
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
        total_rewards += run_episode(env, policy)  # , T=episodes)
    return total_rewards / episodes


def crossover(first, second):
    """
    Pairs from the selected parents are merged to
     generate new offspring child solution.

     Crossover can happen in different forms, simplest
     form is the one-point crossover which splits the
     string representation of each solutions into two
     parts at the same position, then concatenate the
     first part of one solution with the second part
     of the second one to form the offspring solution
     representation.

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
    In biology, mutation happens with low
    probability where a child can have a
    feature that was not inherited from the parents.

    Likewise, in genetic algorithm mutation step
    perturbs the offspring solution with very
    small probability.

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
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    # seed random numbers
    np.random.seed(0)
    env.seed(0)
    # Hyperparameters
    n_policies = 500
    n_generations = 20
    n_fittest = 5
    # initial random policy (population)
    policies = [gen_policy() for _ in range(n_policies)]
    # Genetic Algorithm: loop through generation
    start = time.time()
    for gen in range(n_generations):
        # 1. Evaluate: the current generation (population)
        scores = [eval_policy(env, p) for p in policies]
        print(f'Generation {gen+1:,} Max score = {max(scores):.2f}')
        # 2. Selection: Select n_fittest best policies
        rank_idx = list(reversed(np.argsort(scores)))  # from highest to smallest
        fittest = [policies[idx] for idx in rank_idx[:n_fittest]]
        selection_prob = np.array(scores) / np.sum(scores)  # Convert into probability
        # 3.Crossover: (returns n_policy - n_fittest) crossed-over offsprings
        offsprings = [crossover(policies[np.random.choice(range(n_policies), p=selection_prob)],
                                policies[np.random.choice(range(n_policies), p=selection_prob)])
                      for _ in range(n_policies - n_fittest)]
        # 4. Mutation:
        mutated = [mutation(offspring) for offspring in offsprings]
        # Update the population and add on crossed-over/mutated offsprings
        policies = fittest
        policies += mutated
    time_taken = time.time() - start
    # Evaluate the best policy after crossover & mutation
    scores = [eval_policy(env, policy) for policy in policies]
    best_score = max(scores)
    best_policy = policies[scores.index(best_score)]
    print(f'\nBest policy score = {max(scores):.2f}. Time taken: {time_taken:4.4f}')
    run_episode(env, best_policy, render=True)
