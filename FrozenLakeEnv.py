from Environment import Environment
import contextlib
import numpy as np

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)

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
                # If current state is goal/hole/absorbing state, any action will lead to absorbing state.
                if state in self.goal or state in self.holes or state == self.absorbing_state:
                    self.transaction_probabilties[self.absorbing_state,state,action] = 1
                # Else calculate the probability according to the slip probability
                else:
                    self.transaction_probabilties[next_state,state,action] += 1 - self.slip 
                    for i in range(self.n_actions):
                        self.transaction_probabilties[next_state, state, i] += (self.slip/self.n_actions)

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
