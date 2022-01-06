import NonTabularModelFreeAlgo
import TablularModelFreeAlgo
import ModelBasedAlgo
import FrozenLakeEnv
from LinearWrapper import LinearWrapper

def main():
    seed = 0
    
    lake = [['&', '.', '.', '.','.', '.', '.', '.'],
            ['.', '.', '.', '.','.', '.', '.', '.'],
            ['.', '.', '.', '#','.', '.', '.', '.'],
            ['.', '.', '.', '.','.', '#', '.', '.'],
            ['.', '.', '.', '#','.', '.', '.', '.'],
            ['.', '#', '#', '.','.', '.', '#', '.'],
            ['.', '#', '.', '.','#', '.', '#', '.'],
            ['.', '.', '.', '#','.', '.', '.', '$'],]

    env = FrozenLakeEnv.FrozenLake(lake, slip=0.1, max_steps=64, seed=seed)

    print(env)
    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100
    
    print('')
    
    print('## Policy iteration')
    policy, value = ModelBasedAlgo.policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)
    
    print('')
    
    print('## Value iteration')
    policy, value = ModelBasedAlgo.value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)
    
    print('')
    
    print('# Model-free algorithms')
    max_episodes = 50000
    eta = 0.5
    epsilon = 0.5

    
    print('')
    
    print('## Sarsa')
    policy, value = TablularModelFreeAlgo.sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    
    print('')
    
    print('## Q-learning')
    policy, value = TablularModelFreeAlgo.q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    
    print('')
    
    linear_env = LinearWrapper(env)
    
    print('## Linear Sarsa')
    
    parameters = NonTabularModelFreeAlgo.linear_sarsa(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
    
    print('')
    
    print('## Linear Q-learning')
    
    parameters = NonTabularModelFreeAlgo.linear_q_learning(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

main()