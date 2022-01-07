import numpy as np

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=float)

    # Iterate for max iterations
    for _ in range(max_iterations):
        delta = 0
        for current_state in range(env.n_states):  # for all the current states in n_states
            current_value = value[current_state]
            # sum of probability * (reward+discount_factor*value)
            # Since the policy is deterministic, we consider best action policy[current_state]
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
    iteration_count = 0
    while improved:
        iteration_count += 1
        # Evaluate policy
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        old_policy = policy
        # Improve policy
        policy = policy_improvement(env, value, gamma)
        # Check if any improvement seen, and set the improved flag to False if no improvement is seen
        if np.array_equal(old_policy, policy):
            improved = False

        # If max iterations count is reached, then break out of the loop
        if iteration_count >= max_iterations:
            break

    print('Policy iteration: iterations count : ' + str(iteration_count))
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
    iter_count = 0
    while max_iterations:
        iter_count += 1
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


    print("Value iteration : iterations count " + str(iter_count))
    return policy, value