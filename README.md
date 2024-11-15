# Q Learning Algorithm


## AIM

To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT

Implement the Q-Learning algorithm to train an agent in the Frozen Lake environment and compare its performance with the Monte Carlo algorithm. The objective is to evaluate the success rates, convergence speed, and optimality of policies learned by each algorithm. Performance will be assessed through metrics like average reward per episode and policy stability.

## Q LEARNING ALGORITHM
Implement a Q-Learning algorithm for an agent navigating a grid environment to reach a goal state while maximizing rewards. The agent will learn to select actions based on its experiences, utilizing an epsilon-greedy strategy to balance exploration and exploitation. The performance will be evaluated by the agent's efficiency in reaching the goal over multiple episodes, with insights gained on the impact of hyperparameters on learning outcomes. The ultimate goal is to derive an optimal policy for navigating the grid effectively.

## Q LEARNING FUNCTION
### Name: PRAVEEN S
### Register Number: 212222240077

```
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    
    # Handy variables
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    
    # Q-function and tracking variable for offline analysis
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    
    # Epsilon-greedy action-selection strategy
    select_action = lambda state, Q, epsilon: \
        np.argmax(Q[state]) if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))
    
    # Decay schedule for alpha and epsilon
    alphas = decay_schedule(
        init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(
        init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    
    # Iterating over episodes
    for e in tqdm(range(n_episodes), leave=False):
        # Reset environment
        state, done = env.reset(), False
        
        # Interaction loop for online learning
        while not done:
            # Select an action
            action = select_action(state, Q, epsilons[e])
            next_state, reward, done, _ = env.step(action)
            
            # Full experience tuple (s, a, s', r, d)
            # Update code continues...
            # Update code continues...
                # Calculate TD target
            td_target = reward + gamma * np.max(Q[next_state]) * (not done)

                # Calculate TD error
            td_error = td_target - Q[state][action]

                # Update Q-value
            Q[state][action] += alphas[e] * td_error

                # Update state
            state = next_state

                # Track Q-function and policy
        Q_track[e] = Q.copy()
        pi_track.append(np.argmax(Q, axis=1))

    # Final policy
    V = np.max(Q, axis=1)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
### FVMC:
<img width="324" alt="{B8B7B975-590F-412F-9001-9EF0020972DE}" src="https://github.com/user-attachments/assets/3b951d50-59d4-415c-be48-b3f663462ea7">
<img width="868" alt="{B8C37C8E-99B5-4377-8D83-88254B49CE22}" src="https://github.com/user-attachments/assets/d0a3dfa4-6edd-4d96-b658-bc808141414f">

### Qlearning
<img width="333" alt="{37F3E59B-0BE7-4810-8581-819AC0A990C2}" src="https://github.com/user-attachments/assets/6c371be2-3fad-4e3e-a104-4186d876188e">
<img width="902" alt="{DCBA3565-DBE8-4F98-90A9-D2A048113C18}" src="https://github.com/user-attachments/assets/5130570f-c696-4e7e-b279-c43008143638">


## RESULT:

The Q-Learning algorithm demonstrated a faster convergence compared to the Monte Carlo algorithm in the Frozen Lake environment, achieving optimal policies more effectively. Both algorithms provided insights into the influence of hyperparameters, with Q-Learning proving to be more stable in policy performance over time.
