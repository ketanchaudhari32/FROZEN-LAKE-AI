import numpy as np

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
