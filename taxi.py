"""
Link: http://gym.openai.com/
"""
import time
import gym

env = gym.make("Taxi-v2")
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    print("Observation", observation)
    print("Reward", reward)
    print("Done", done)
    print("Info", info)
    # break
    time.sleep(1)
