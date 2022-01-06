################ Environment ################

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

#importing probaabilty data
#prob = np.load('p.npy')
#print(prob)

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
            self.pi = np.full(n_states, 1./n_states)
        
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
    def __init__(self, lake, slip, max_steps, seed=None):
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
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        
        self.slip = slip
        
        self.n_states = self.lake.size + 1
        self.n_actions = 4
        
        self.pi = np.zeros(self.n_states, dtype=float)
        self.pi[np.where(self.lake_flat == '&')[0]] = 1.0
        
        self.absorbing_state = self.n_states - 1
        

        self.max_steps = max_steps
        self.random_state = np.random.RandomState(seed)

        self.holes = np.where(self.lake_flat == "#")[0]
        self.goal = np.where(self.lake_flat == "$")[0]

        # Set the probabilities for each state and action
        self.transaction_probabilties = np.zeros((self.n_states,self.n_states,self.n_actions))
        for state in range(self.n_states):
            for action in range(self.n_actions):
                # Get the valid next state
                next_state = self.get_valid_next_state(state,action)
                # If current state is goal state, any action will lead to absorbing state.
                if state in self.goal:
                    self.transaction_probabilties[self.absorbing_state,state,action] = 1
                # Else calculate the probability according to the slip probability
                else:
                    self.transaction_probabilties[next_state,state,action] += 1 - self.slip 
                    for i in range(self.n_actions):
                        self.transaction_probabilties[next_state, state, i] += (self.slip/self.n_actions)

    # Find the next valid state based on any action up, left, down or right
    # If an action leads to going out of the grid, then return the current state itself, otherwise return the appropriate next state
    def get_valid_next_state(self, state, action):
        if action == 0 and state < self.n_states - 1  and state not in self.holes: #up
            if state - self.lake.shape[0] >= 0:
                return state - self.lake.shape[0] 
            else:
                return state

        if action == 1 and state < self.n_states - 1  and state not in self.holes: #left
            if state % self.lake.shape[0] != 0:
                return state - 1
            else:
                return state

        if action == 2 and state < self.n_states - 1  and state not in self.holes: #down
            if state + self.lake.shape[0] < self.n_states - 1:
                return state + self.lake.shape[0] 
            else:
                return state

        if action == 3 and state < self.n_states - 1  and state not in self.holes: #right
            if (state+1) % self.lake.shape[0] != 0:
                return state + 1
            else:
                return state

        return state

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done
        
    def p(self, next_state, state, action):
        # Return the probability of reaching next_state from state based on an action
        return self.transaction_probabilties[next_state,state,action]
    
    def r(self, next_state, state, action):
         # If the current state is goal state, return reward 1, otherwise return reward 0
        if state in self.goal:
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
    value = np.zeros(env.n_states, dtype=float)

    # Iterate for max iterations
    for _ in range(max_iterations):
        delta = 0
        for current_state in range(env.n_states):  # for all the current states in n_states
            current_value = value[current_state]
            # sum of probability * (reward+discount_factor*value)
            value[current_state] = sum([env.p(next_state, current_state, policy[current_state]) * (env.r(next_state, current_state, policy[current_state]) + gamma * value[next_state]) for next_state in range(env.n_states)])
            # delta will get the maximum value between current delta or measure of change in values
            delta = max(delta, abs(current_value - value[current_state]))

        # Stop if delta is less than a certain tolerance theta
        if delta < theta:
            break
    return value
    
def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    
    for current_state in range(env.n_states):
        # take sum of p(r+gamma*value) for all possible next states
        action_values = [np.sum([env.p(next_state, current_state, action) * (env.r(next_state, current_state, action) + gamma * value[next_state]) for next_state in range(env.n_states)]) for action in range(env.n_actions)]
        # take the argument of the max value for all possible actions
        policy[current_state] = np.argmax(action_values)

    # Return the improved policy
    return policy
    
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    # Implement the policy iteration logic
    improved = True
    value = None
    while improved:
        # Evaluate policy
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        old_policy = policy
        # Improve policy
        policy = policy_improvement(env, value, gamma)
        # Check if any improvement seen, and set the improved flag to False if no improvement is seen
        if np.array_equal(old_policy, policy):
            improved = False

    # Return optimal policy, and value
    return policy, value
    
def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    
    # Implement the core logic for value iteration
    policy = np.zeros(env.n_states, dtype=int)
    # Loop till max iterations reached
    while max_iterations:
        # Set delta to 0 for current iteration
        delta = 0
        for current_state in range(env.n_states):  # for all the current states in n_states
            current_value = value[current_state]
            # sum of probability * (reward+discount_factor*value)
            action_values = [np.sum([env.p(next_state, current_state, action) * (env.r(next_state, current_state, action) + gamma * value[next_state]) for next_state in range(env.n_states)]) for action in range(env.n_actions)]
            value[current_state] = np.max(action_values)
            # delta will get the maximum value between current delta or measure of change in values
            delta = max(delta, abs(current_value - value[current_state]))
        # Stop if delta is less than a certain tolerance theta
        if delta < theta:
            break

        max_iterations -= 1

    # Derive the optimal policy
    for current_state in range(env.n_states):
        action_values = [np.sum([env.p(next_state, current_state, action) * (env.r(next_state, current_state, action) + gamma * value[next_state]) for next_state in range(env.n_states)]) for action in range(env.n_actions)]
        policy[current_state] = np.argmax(action_values)

    return policy, value

################ Tabular model-free algorithms ################

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()


        # Use e-greedy policy
        if epsilon[i] > np.random.rand(1)[0]:
            # Choose random action for exploration
            curr_action = np.random.choice(env.n_actions)
        else:
            # Choose best action (break ties randomly)
            curr_action=np.random.choice(np.flatnonzero(q[s] == q[s].max()))

        done = False
        while not done:

            # Get the next state, reward and done using step function
            next_state, next_reward, done = env.step(curr_action)  
            # Use e-greedy policy
            if epsilon[i] > np.random.rand(1)[0]:
                # Choose random action for exploration
                best_action = np.random.choice(env.n_actions)
            else:
                # Choose best action (break ties randomly)
                best_action = np.random.choice(np.flatnonzero(q[next_state] == q[next_state].max()))

            # Updating q values
            q[s,curr_action] = q[s,curr_action] + eta[i] * (next_reward + gamma * q[next_state,best_action] - q[s,curr_action])

            # Updating current state and action
            s = next_state
            curr_action = best_action

            
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value
    
def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))

    # Iterate for max episodes
    for i in range(max_episodes):
        s = env.reset()

        done = False
        while not done:

            # Use e-greedy policy
            if epsilon[i] > np.random.rand(1)[0]:
                # Choose random action for exploration
                curr_action = np.random.choice(env.n_actions)
            else:
                # Choose best action (break ties randomly)
                curr_action = np.random.choice(np.flatnonzero(q[s] == q[s].max()))

            # Get the next state, reward and done using step function
            next_state, next_reward, done = env.step(curr_action)   

            # Get the best action
            best_action = np.argmax(q[next_state])

            # Updating q values
            q[s,curr_action] = q[s,curr_action] + eta[i] * (next_reward + gamma * q[next_state,best_action] - q[s,curr_action])

            # Updating current state and action
            s = next_state
            
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
    
    for i in range(max_episodes):
        features = env.reset()
        
        q = features.dot(theta)

        # Use e-greedy policy
        if epsilon[i] > np.random.rand(1)[0]:
            # Choose random action for exploration
            curr_action = np.random.choice(env.n_actions)
        else:
            # Choose best action (break ties randomly)
            curr_action = np.random.choice(np.flatnonzero(q == q.max()))

        done = False
        while not done:

            # Get the next feature, reward and done using step function
            next_feature, next_reward, done = env.step(curr_action)

            # Updating delta
            delta = next_reward - q[curr_action]

            # Updating q
            q = next_feature.dot(theta)

            # Use e-greedy policy
            if epsilon[i] > np.random.rand(1)[0]:
                # Choose random action for exploration
                action = np.random.choice(env.n_actions)
            else:
                # Choose best action (break ties randomly)
                action = np.random.choice(np.flatnonzero(q == q.max()))

            # Updating delta and theta
            delta = delta + gamma * q[action]
            theta = theta + eta[i] * delta * features[curr_action]

            features = next_feature
            curr_action = action
    
    return theta
    
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        q = features.dot(theta)

        done = False
        while not done:

            # Use e-greedy policy
            if epsilon[i] > np.random.rand(1)[0]:
                # Choose random action for exploration
                curr_action = np.random.choice(env.n_actions)
            else:
                # Choose best action (break ties randomly)
                curr_action = np.random.choice(np.flatnonzero(q == q.max()))

            # Updating feature variable
            next_feature, next_reward, done = env.step(curr_action)

            # Updating delta
            delta = next_reward - q[curr_action]

            # Updating q
            q = next_feature.dot(theta)

            # Updating delta and theta
            delta = delta + gamma * np.max(q)
            theta = theta + eta[i] * delta * features[curr_action]

            features = next_feature
    
    return theta   

################ Main function ################

def main():
    seed = 0
    
    # Small lake
    lake =   [['&', '.', '.', '.'],
              ['.', '#', '.', '#'],
              ['.', '.', '.', '#'],
              ['#', '.', '.', '$']]

    #big lake
    # lake = [['&', '.', '.', '.','.', '.', '.', '.'],
    #         ['.', '.', '.', '.','.', '.', '.', '.'],
    #         ['.', '.', '.', '#','.', '.', '.', '.'],
    #         ['.', '.', '.', '.','.', '#', '.', '.'],
    #         ['.', '.', '.', '#','.', '.', '.', '.'],
    #         ['.', '#', '#', '.','.', '.', '#', '.'],
    #         ['.', '#', '.', '.','#', '.', '#', '.'],
    #         ['.', '.', '.', '#','.', '.', '.', '$'],]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed) # small lake
    #env = FrozenLake(lake, slip=0.1, max_steps=64, seed=seed) #big lake

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

    # max_episodes = 5000
    # eta = 0.8
    # epsilon = 0.9
    
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