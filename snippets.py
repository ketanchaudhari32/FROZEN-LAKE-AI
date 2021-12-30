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
        
        # TODO:
        self.max_steps = max_steps
        self.random_state = np.random.RandomState(seed)

        self.holes = np.where(self.lake_flat == "#")[0]
        self.goal = np.where(self.lake_flat == "$")[0]
        
        self.transaction_probabilties = np.zeros((self.n_states,self.n_states,self.n_actions))
        for state in range(self.n_states):
            for action in range(self.n_actions):
                next_state = self.get_valid_next_state(state,action)
                if state in self.goal:
                    self.transaction_probabilties[self.absorbing_state,state,action] = 1
                else:
                    self.transaction_probabilties[next_state,state,action] += 1 - self.slip 
                    for i in range(self.n_actions):
                        self.transaction_probabilties[next_state, state, i] += (self.slip/self.n_actions)

    def get_valid_next_state(self, state, action):
        if action == 0 and state < self.n_states - 1  and state not in self.holes:#up
            if state - self.lake.shape[0] >= 0:
                return state - self.lake.shape[0] 
            else:
                return state

        if action == 1 and state < self.n_states - 1  and state not in self.holes:#left
            if state - self.lake.shape[0] >= 0 and state % self.lake.shape[0]!=0:
                return state - 1 
            else:
                return state

        if action == 2 and state < self.n_states - 1  and state not in self.holes:#down
            if state + self.lake.shape[0] < self.n_states - 1:
                return state + self.lake.shape[0] 
            else:
                return state

        if action == 3 and state < self.n_states - 1  and state not in self.holes:#right
            if state + self.lake.shape[0] >= 0 and (state+1) % self.lake.shape[0]!=0:
                return state + 1 
            else:
                return state
        
        return state

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done
        
    def p(self, next_state, state, action):
        # TODO:
        return self.transaction_probabilties[next_state,state,action]
    
    def r(self, next_state, state, action):
        # TODO:
        if next_state in self.goal:
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
    value = np.zeros(env.n_states, dtype=np.float)

    # TODO:
    current_iteration = 0  # initialisaton of iterations

    while current_iteration < max_iterations:
        delta = 0
        for current_state in range(env.n_states):  # for all the current states in n_states
            current_value = value[current_state]
            # sum of probability * (reward+discount_factor*value)
            value[current_state] = sum([env.p(next_state, current_state, policy[current_state]) * (env.r(next_state, current_state, policy[current_state]) + gamma * value[next_state]) for next_state in range(env.n_states)])
            # delta will get the maximum value between current delta or measure of change in values
            delta = max(delta, abs(current_value - value[current_state]))
        if delta < theta:
            break
        current_iteration += 1

    return value
    
def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    
    # TODO:
    for s in range(env.n_states):
        # take sum of p(r+gamma*value) for all possible next states
        action_values = [sum([env.p(next_state, s, action) * (env.r(next_state, s, action) + gamma * value[next_state]) for next_state in range(env.n_states)]) for action in range(env.n_actions)]
        # take the argument of the max value for all possible actions
        policy[s] = np.argmax(action_values)

    return policy
    
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    # TODO:
    iterations = 0
    stable = False
    while not stable:
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy, stable = policy_improvement(env, policy, value, gamma)
        iterations = iterations+1  
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

    for i in range(max_episodes):
        s = env.reset()
        # TODO:

        #e-greedy policy
        if epsilon[i] > np.random.rand() or max(q[s]) == 0:
            curr_action = np.random.choice(env.n_actions)
        else:
            curr_action= np.argmax(q[s]) 


        done = False
        while not done:

            #updating state variable
            next_state, next_reward, done = env.step(curr_action)  

            #e-greedy policy
            if epsilon[i] > np.random.rand() or max(q[s]) == 0:
                best_action = np.random.choice(env.n_actions)
            else:
                best_action = np.argmax(q[next_state])   

            #updating q values
            q[s,curr_action] = q[s,curr_action] + eta[i] * (next_reward + gamma * q[next_state,best_action] - q[s,curr_action])

            #updating current state and action
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
    
    for i in range(max_episodes):
        s = env.reset()
        # TODO:

        done = False
        while not done:

            #e-greedy policy
            if epsilon[i] > np.random.rand() or max(q[s]) == 0:
                curr_action = np.random.choice(env.n_actions)
            else:
                curr_action = np.argmax(q[s])  

            #updating state variable
            next_state, next_reward, done = env.step(curr_action)   

            best_action = np.argmax(q[next_state])

            #updating q values
            q[s,curr_action] = q[s,curr_action] + eta[i] * (next_reward + gamma * q[next_state,best_action] - q[s,curr_action])

            #updating current state and action
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

        # TODO:
        
        #e-greedy policy
        if 1 - epsilon[i] > np.random.rand() or max(q) == 0:
            curr_action = np.random.choice(env.n_actions)
        else:
            curr_action= np.argmax(q) 


        done = False
        while not done:

            #updating feature variable
            next_feature, next_reward, done = env.step(curr_action)

            #updating delta
            delta = next_reward - q[curr_action]

            #updating q
            q = next_feature.dot(theta)

            #e-greedy policy
            if 1 - epsilon[i] > np.random.rand():
                best_action = np.random.choice(env.n_actions)
            else:
                best_action = np.argmax(q)      

            #updating delta and theta
            delta = delta + gamma * q[best_action]
            theta = theta + eta[i] * delta * features[curr_action]

            features = next_feature
            curr_action = best_action
    
    return theta
    
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        q = features.dot(theta)

        # TODO:
        
        done = False
        while not done:

            #e-greedy policy
            if epsilon[i] > np.random.rand(1)[0] or max(q) == 0:
                curr_action = np.random.choice(env.n_actions)
            else:
                curr_action = np.argmax(q)      

            #updating feature variable
            next_feature, next_reward, done = env.step(curr_action)

            #updating delta
            delta = next_reward - q[curr_action]

            #updating q
            q = next_feature.dot(theta)

            #updating delta and theta
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