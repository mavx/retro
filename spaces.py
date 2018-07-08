import gym

env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

from gym import envs
all_envs = envs.registry.all()
print(all_envs)
