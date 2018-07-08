import gym
# env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
env = gym.make('MsPacman-v0')
env.reset()
for _ in range(1000):
    env.render()
    step = env.step(env.action_space.sample()) # take a random action
    print(step)
    for tup in step:
        print(tup.shape)
    break
