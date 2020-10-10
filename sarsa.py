import gym
import numpy

############
# Parameters
############
discount_factor = 0.05
exploration_chance = 0.05
step_size = 0.5
steps_per_episode = 200
training_episodes = 100
# Tile configuration
position_levels = 20
velocity_levels = 20

actions = [0, 1, 2]
# Q[tile, action]
Q = [[0 for _ in actions] for _ in range(position_levels*velocity_levels)] # Optimistic for now

env = gym.make('MountainCar-v0')

position_bins = numpy.linspace(env.observation_space.low[0], env.observation_space.high[0], position_levels)
velocity_bins = numpy.linspace(env.observation_space.low[1], env.observation_space.high[1], velocity_levels)
def discretize(observation: numpy.ndarray) -> numpy.int64:
    """Convert a continuous position and velocity to a discrete tile."""
    return numpy.digitize(observation[0], position_bins) * velocity_levels + numpy.digitize(observation[1], velocity_bins)

rand = numpy.random.default_rng()
def policy(state: int) -> numpy.int64:
    """Given a state, return the action to take based on Q."""
    # Exploration
    if rand.random() < exploration_chance:
        return rand.choice(actions)
    # or nah
    return numpy.argmax(Q[state])

# Loop for each episode:
for episode in range(training_episodes):
    # Initialize S
    observation = env.reset()
    state = discretize(observation)

    # Choose A from S using policy derived from Q
    action = policy(state)

    # Loop for each step of episode:
    t = 0
    total_reward = 0
    for _ in range(steps_per_episode):
        # Take action A, observe R, S'
        t += 1
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            break
        next_state = discretize(observation)
        total_reward += reward

        # Choose A' from S' using policy derived from Q
        next_action = policy(next_state)

        # Update Q
        Q[state][action] += step_size * (reward + discount_factor*Q[next_state][next_action] - Q[state][action])

        # Update S, A
        state = next_state
        action = next_action
    
    print('Episode %d: Reward %d after %d steps' % (episode, total_reward, t))
