"""Author: James Martindale
   Date: 10/12/20
"""

import gym
from statistics import mean
import numpy
from tqdm import tqdm

def sarsa_experiment(steps_per_episode:int=200, training_episodes:int=5000,
                position_levels:int=20, velocity_levels:int=20,
                learning_rate:float=0.1, exploration_chance:float=0.05,
                discount_factor:float=0.9, render:bool=False):
    """Returns list of episode scores, first successful episode, best score, percentage successful episodes."""
    # Override episode length
    custom_env = 'MountainCar-CustomLength-v0'
    gym.envs.registry.env_specs.pop(custom_env, None) # For persistent kernels
    gym.envs.register(
        id=custom_env,
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
    def policy(state: int) -> numpy.int64:
        """Given a state, return the action to take based on Q."""
        # Exploration
        if rand.random() < exploration_chance:
            return rand.choice(actions)
        # or nah
        return numpy.argmax(Q[state])

    # Loop for each episode:
    episode_scores = []
    first_success = None
    for episode in tqdm(range(training_episodes), unit=' trials'):
        # Initialize S
        observation = env.reset()
        state = tile(observation)

        # Choose A from S using policy derived from Q
        action = policy(state)

        # Loop for each step of episode:
        t = 0
        total_reward = 0
        done = False
        while not done:
            # Take action A, observe R, S'
            t += 1
            observation, reward, done, info = env.step(action)
            if render:
                env.render()
            next_state = tile(observation)
            total_reward += reward

            # Choose A' from S' using policy derived from Q
            next_action = policy(next_state)

            # Update Q
            Q[state][action] += learning_rate * (reward + discount_factor*Q[next_state][next_action] - Q[state][action])

            # Update S, A
            state = next_state
            action = next_action
        
        if not first_success and total_reward > -200:
            first_success = episode
        episode_scores.append(total_reward)
    
    # Return trial info
    return (episode_scores, first_success, max(episode_scores), len([score for score in episode_scores if score > -200])/training_episodes)