import gym
import numpy

############
# Parameters
############
discount_factor = 0.9
alpha = 0.3
exploration_chance = 0.05
render = False
step_size = 0.5
# Originally 200
steps_per_episode = 300
training_episodes = 500
# Tile configuration
position_levels = 20
velocity_levels = 20

# Override episode length
gym.envs.register(
    id='MountainCar-CustomLength-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=steps_per_episode,
    reward_threshold=-110.0,
)

actions = [0, 1, 2]
# Q[tile, action]
Q = [[0 for _ in actions] for _ in range(position_levels*velocity_levels)] # Optimistic for now

env = gym.make('MountainCar-CustomLength-v0')

position_bins = numpy.linspace(env.observation_space.low[0], env.observation_space.high[0], position_levels)
velocity_bins = numpy.linspace(env.observation_space.low[1], env.observation_space.high[1], velocity_levels)
def tile(observation: numpy.ndarray) -> numpy.int64:
    """Convert a continuous position and velocity to a discrete tile."""
    return numpy.digitize(observation[0], position_bins) * velocity_levels + numpy.digitize(observation[1], velocity_bins)

rand = numpy.random.default_rng()
def policy(state):
    if rand.random() < exploration_chance:
        return rand.choice(actions)
    else:
        return numpy.argmax(Q[state])

def update_rule(state, action, next_state, next_action, reward):
    Q[state][action] += Q[state][action] + alpha * (reward + discount_factor * Q[next_state][next_action] - Q[state][action])

for episode in range(training_episodes):
    observation = env.reset()
    state = tile(observation)
    action = policy(state)

    step = 0
    total_reward = 0
    done = False

    while not done:
        step += 1
        observation, reward, done, info = env.step(action)

        next_state = tile(observation)
        next_action = policy(next_state)
        total_reward += reward
        update_rule(state, action, next_state, next_action, reward)

        state = next_state
        action = next_action
    print('Episode %d: Reward %d after %d steps' % (episode, total_reward, step))