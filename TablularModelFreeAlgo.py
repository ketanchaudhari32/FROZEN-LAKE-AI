import numpy as np

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
