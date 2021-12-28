################ Environment ################
from itertools import product

import numpy as np
import contextlib


# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


# importing probaabilty data
# prob = np.load('p.npy')
# print(prob.shape)

class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1. / n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None, action=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
              ['.', '#', '.', '#'],
              ['.', '.', '.', '#'],
              ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        pnpy = np.load('p.npy')
        print(pnpy)
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip

        self.n_states = self.lake.size + 1
        self.n_actions = 4

        pi = np.zeros(self.n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        self.absorbing_state = self.n_states - 1

        # TODO:
        # call parent constructor
        Environment.__init__(self, self.n_states, self.n_actions, max_steps, pi, seed)
        self.actions = ['0', '1', '2', '3']
        # Matrix with lake shape initialized
        self.reward_map = np.zeros(self.lake.shape, dtype=np.float)
        # goal state setting to 1
        self.reward_map[np.where(self.lake == '$')] = 1
        # Matrix with absorbing state initialized.
        self.absorbing_state = np.zeros(self.lake.shape, dtype=np.float)
        # Set goal state and holes as 1
        self.absorbing_state[np.where(self.lake == '$')] = 1
        self.absorbing_state[np.where(self.lake == '#')] = 1

        self.index_to_coordinates = list(product(range(self.reward_map.shape[0]), range(self.reward_map.shape[1])))
        self.coordinates_to_index = {s: i for (i, s) in enumerate(self.index_to_coordinates)}

        hole = np.where(self.lake_flat == "#")[0]
        self.goal = np.where(self.lake_flat == "$")

        # Checks to keep agent in the grid, so that it doesn't leave the grid
        def movement(row, column, action):
            # upward movement
            if action == 0:
                row = max(row - 1, 0)
            # left movement
            elif action == 1:
                column = max(column - 1, 0)
            # downward movement
            elif action == 2:
                row = min(row + 1, self.lake.shape[0] - 1)
            # Right movement
            elif action == 3:
                column = min(column + 1, self.lake.shape[1] - 1)
            return row, column

        def new_matrix(row, column, action):

            # grid check for next grid coordinates
            new_row, new_column = movement(row, column, action)
            # Change coordinates to index
            state_next = self.coordinates_to_index[(new_row, new_column)]
            # Get new coordinate type
            coordinate_type = self.lake[row][column]
            # Check whether we reached the goal or fell in hole
            done = coordinate_type == '$' or coordinate_type == '#'
            # if agent is at goal ($), reward = 1 else it is  0
            reward = float(coordinate_type == '$')
            return state_next, reward, done

        for state in range(self.n_states):
            if state == self.absorbing_state or state in hole or state in self.goal:
                self.transitioning_probability[self.absorbing_state, state, action] = 1






    def step(self, action):
        state, reward, done = Environment.step(self, action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):
        # initialize transitioning probability
        self.transitioning_probability = np.zeros((self.n_states, self.n_states, self.n_actions))


        # TODO:
        if state == next_state or self.lake[state][next_state] == '#':
            return 0
        elif self.lake[state][next_state] == '$':
            return 1
        else:
            return 0.5

    def r(self, next_state, state, action):
        # TODO:
        if self.lake[state][next_state] == '$':
            return 1
        else:
            return 0

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))


def play(env):
    actions = ['w', 'a', 's', 'd']

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')

        state, r, done = env.step(actions.index(c))

        env.render()
        print('Reward: {0}.'.format(r))


################ Model-based algorithms ################

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float) # create array size of n_states of type numpy float
    # TODO:
    current_iteration = 0  # initialisaton of iterations

    while current_iteration < max_iterations:
        delta = 0
        for current_state in range(env.n_states):  # for all the current states in n_states
            current_value = value[current_state]
            # sum of probability * (reward+discount_factor*value)
            value[current_state] = sum([env.p(next_state, current_state, policy[current_state]) * (env.r(next_state, current_state, policy[current_state]) + gamma * value[next_state]) for next_state in
                             range(env.n_states)])
            # delta will get the maximum value between current delta or measure of change in values
            delta = max(delta, abs(current_value - value[current_state]))
        if delta < theta:
            break
        current_iteration += 1

    return value


def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)

    # TODO:

    return policy


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    # TODO:
    value = np.zeros(env.n_states)
    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    # TODO:
    policy = np.zeros(env.n_states, dtype=int)
    return policy, value


################ Tabular model-free algorithms ################

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    actions = [0, 1, 2, 3]
    """"
    0=up
    1=left
    s=down
    d=right
    """
    best_action = None

    for i in range(max_episodes):
        s = env.reset()
        # TODO:

        # e-greedy policy
        if epsilon[i] > np.random.rand(1)[0]:
            best_action = np.random(actions)
        else:
            best_action = np.argmax(env.p(s, random_state, actions[i]) for i in actions)

        state1, reward1, done = env.step(best_action)
        while done:

            # e-greedy policy
            if epsilon[i] > np.random.rand(1)[0]:
                best_action = np.random(actions)
            else:
                best_action = np.argmax(env.p(s, random_state, actions[i]) for i in actions)

                # updating q values
            for j in range(env.n_states):
                for k in range(env.n_actions):
                    q[j][k] = q[j][k] + eta[i] * (reward1 + gamma * q[state1][best_action] - q[j][k])

            # updating state variable
            state1, reward1, done = env.step(best_action)

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    actions = [0, 1, 2, 3]
    """"
    0=up
    1=left
    2=down
    3=right
    """
    action = None
    done = False

    for i in range(max_episodes):
        s = env.reset()
        # TODO:

        while not done:
            # e-greedy policy
            if epsilon[i] > np.random.rand(1)[0]:
                action = np.random(actions)
            else:
                action = np.argmax(env.p(s, random_state, actions[i]) for i in actions)

            # Get the observations after taking the action
            state1, reward1, done = env.step(action)

            best_next_action = np.argmax(q[state1])
            # Updating q value
            q[s][action] = q[s][action] + eta[i] * (reward1 + gamma * q[state1][best_next_action] - q[s][action])

            state = state1

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


################ Non-tabular model-free algorithms ################

class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    actions = [0, 1, 2, 3]
    """"
    0=up
    1=left
    s=down
    d=right
    """
    best_action = None

    for i in range(max_episodes):
        features = env.reset()

        q = features.dot(theta)

        # TODO:
        # e-greedy policy
        if epsilon[i] > np.random.rand(1)[0]:
            best_action = np.random(actions)
        else:
            best_action = np.argmax(env.p(s, random_state, actions[i]) for i in actions)

        state1, reward1, done = env.step(best_action)
        while done:

            # e-greedy policy
            if epsilon[i] > np.random.rand(1)[0]:
                best_action = np.random(actions)
            else:
                best_action = np.argmax(env.p(s, random_state, actions[i]) for i in actions)

                # updating delta
            delta = reward1 - q(best_action)

            # updating q
            q = state1.dot(theta)

            # updating delta and theta
            delta = delta + gamma * q[best_action]

            theta[i] = theta[i] + eta[i] * delta * state1

            # updating state variable
            state1, reward1, done = env.step(best_action)

    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()

        # TODO:

    return theta


################ Main function ################

def main():
    seed = 0

    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    print(env)
    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('# Model-free algorithms')
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5

    print('')

    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')

    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')

    linear_env = LinearWrapper(env)

    print('## Linear Sarsa')

    parameters = linear_sarsa(linear_env, max_episodes, eta,
                              gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')

    print('## Linear Q-learning')

    parameters = linear_q_learning(linear_env, max_episodes, eta,
                                   gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)


main()
