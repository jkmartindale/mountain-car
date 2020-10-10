"""Does nothing useful, just sets up the environment and runs a single episode taking random actions."""

# Preamble
import gym
env = gym.make('MountainCar-v0')
env.reset()

# Single episode
t = 0
for _ in range(1000): # Cap at 1000 steps
    env.render()
    t += 1
    observation, reward, done, info = env.step(env.action_space.sample()) # Random action
    if done:
        break

env.close()
print('Episode finished after %s steps' % t)
