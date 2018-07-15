from gym import envs

for e in envs.registry.all():
    env = e
    print(env)

print("DONE")
