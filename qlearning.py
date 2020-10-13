import gym
import numpy
import tqdm


class qlearningModel:
    ############
    # Parameters
    ############
    def __init__(self, discount_factor, alpha, training_episode, pl, vl, render=False, headless=False):
        self.headless = headless
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.exploration_chance = 0.05
        self.render = render
        # Originally 200
        self.training_episode = training_episode
        # Tile configuration
        self.position_levels = pl
        self.velocity_levels = vl
        self.actions = [0, 1, 2]
        # Q[tile, action]
        self.Q = [[0 for _ in self.actions] for _ in range(self.position_levels * self.velocity_levels)]  # Optimistic for now
        gym.envs.register(
            id='MountainCar-CustomLength-v0',
            entry_point='gym.envs.classic_control:MountainCarEnv',
            max_episode_steps=1000,
            reward_threshold=-110.0,
        )
        self.env = gym.make('MountainCar-CustomLength-v0')
        self.position_bins = numpy.linspace(self.env.observation_space.low[0], self.env.observation_space.high[0],
                                       self.position_levels)
        self.velocity_bins = numpy.linspace(self.env.observation_space.low[1], self.env.observation_space.high[1],
                                       self.velocity_levels)
        self.rand = numpy.random.default_rng()

    # Override episode length

    def resetQ(self):
        self.Q = [[0 for _ in self.actions] for _ in range(self.position_levels * self.velocity_levels)]

    def tile(self, observation: numpy.ndarray) -> numpy.int64:
        """Convert a continuous position and velocity to a discrete tile."""
        return numpy.digitize(observation[0], self.position_bins) * self.velocity_levels + numpy.digitize(observation[1], self.velocity_bins)

    def policy(self, state):
        if self.rand.random() < self.exploration_chance:
            return self.rand.choice(self.actions)
        else:
            #if len(set(Q[state])) == 1:
             #   return numpy.random.randint(0,2)
            return numpy.argmax(self.Q[state])

    def update_rule(self, state, action, next_state, next_action, reward):
        update_value = self.Q[state][action] + self.alpha * (reward + self.discount_factor * self.Q[next_state][next_action] -
                                                        self.Q[state][action])
        self.Q[state][action] = update_value

    def run(self, max_steps):
        good_runs = 0
        first_under = 0
        best_score = -99999
        self.env._max_episode_steps = max_steps
        self.resetQ()
        print("Running model:")
        reward_values = []
        for episode in tqdm.trange(self.training_episode):
            observation = self.env.reset()
            state = self.tile(observation)
            action = self.policy(state)
            step = 0
            total_reward = 0
            done = False
            #if episode % 5 == 0 and exploration_chance > .05:
             #   exploration_chance -= 0.05
            if episode == 4500:
                self.env.render()
            while not done:
                step += 1
                observation, reward, done, info = self.env.step(action)

                next_state = self.tile(observation)
                next_action = self.policy(next_state)
                total_reward += reward
                self.update_rule(state, action, next_state, next_action, reward)

                state = next_state
                action = next_action
            reward_values.append(total_reward)
            if self.headless:
                print('Episode %d: Reward %d after %d steps' % (episode, total_reward, step))
            if total_reward > best_score:
                best_score = total_reward
            if step < 200:
                good_runs += 1
                if first_under == 0:
                    first_under = episode

        print("Runs under 200 steps: %d" % good_runs)
        return reward_values, first_under, best_score, good_runs