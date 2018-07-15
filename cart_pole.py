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
            # print("Episode finished after {} timesteps".format(t))
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
    reward = 0
    if params is not None:
        print("Testing params: {}".format(params))
        reward = run_game(params)
        print("Test reward: {}\n".format(reward))

    return reward

def train_once():
    print("##### TRAINING ONCE #####")
    params = train()
    reward = test(params)

    if reward < 200:
        print("FAILED! Final test failed at {} with params {}".format(reward, params))
    else:
        print("PASS. Final test passed at {} with params {}".format(reward, params))

def train_twice():
    print("##### TRAINING TWICE #####")
    best_params = None
    max_reward = 0.0
    for _ in range(100):
        params = train()
        reward = test(params)
        if reward > max_reward:
            print("Found better params {} with reward {}".format(params, reward))
            max_reward = reward
            best_params = params

    reward = test(best_params)
    print("Final test 1: Rewarded {} with params {}\n\n".format(reward, best_params))

    reward = test(best_params)
    print("Final test 2: Rewarded {} with params {}\n\n".format(reward, best_params))

    reward = test(best_params)
    print("Final test 3: Rewarded {} with params {}\n\n".format(reward, best_params))

def main():
    train_once()
    train_twice()

if __name__ == '__main__':
    main()
