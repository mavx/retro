"""
Utility functions mainly for debugging
"""
import time
import json
import sys

import gym
from gym import envs

def print_games():
    for n, e in enumerate(envs.registry.all()):
        print(n, e.id)

def update_game_list():
    filename = 'games.json'
    games = (e.id for e in envs.registry.all())
    game_index = {n: g for n, g in enumerate(sorted(games))}

    with open(filename, 'w') as o:
        o.write(json.dumps(game_index, indent=2, sort_keys=True))
    print("Updated game list: {}".format(filename))

def countdown(secs):
    print("Counting down {} seconds..".format(secs))
    while secs > 0:
        print(secs)
        time.sleep(1)
        secs -= 1

def run_game(game_name):
    env = gym.make(game_name)
    print("Running: {} for 20 episodes".format(str(env)))
    print("Action Space: {}".format(env.action_space))
    print("Observation Space: {}".format(env.observation_space))
    
    countdown(3)
    try:
        for i_episode in range(20):
            print("############# Episode {} #############".format(i_episode))
            observation = env.reset()
            for t in range(1000):
                print("\n############# Round {} #############".format(t))
                env.render()
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                print("Observation: {}".format(observation))
                print("Reward: {}".format(reward))
                print("Done: {}".format(reward))
                print("Info: {}".format(info))
                time.sleep(0.2)

                if done:
                    print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                    countdown(5)
                    break
    except KeyboardInterrupt as e:
        print("Interrupted manually")
    finally:
        print("Completed")

def main():
    if len(sys.argv) < 2:
        print("Please specify argument, e.g:")
        print("python helper.py update_game_list")
        return

    task = sys.argv[1]
    if task == 'update_game_list':
        update_game_list()
    elif task == 'print_games':
        print_games()
    elif task == 'run':
        if (len(sys.argv) < 3):
            print("Please specify game name. Arguments: {}".format(sys.argv))
            return
        game = sys.argv[2]
        run_game(game)
    else:
        print("Unknown task")


if __name__ == '__main__':
    main()
