"""
Strategy: 
"""

import gym
import time
import numpy as np

GAME = gym.make('CartPole-v1')

def run_game(np_params):
    observation = GAME.reset()
    done = False
    t = 0
    total_reward = 0
    while not done:
        t += 1
        decision = np.matmul(np_params, observation)
        action = 0 if decision < 0 else 1
        observation, reward, done, info = GAME.step(action)
        total_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t))
            break
    
    return total_reward

def train():
    for t in range(10000):
        init_params = np.random.rand(4) * 2 - 1 # Initializing between [-1, 1]
        reward = run_game(init_params)
        if reward == 200:
            print("Achieved 200 at time {} with params {}".format(t, init_params))
            return init_params

def test(params):
    if params is not None:
        print("Playing in the finals!")
        reward = run_game(params)
        print("Final reward:", reward)
    else:
        print("Failed, no params:", params)

def main():
    params = train()
    test(params)


if __name__ == '__main__':
    main()
