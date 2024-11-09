import numpy as np
import random

# Define the grid world
grid_size = 5
actions = ['up', 'down', 'left', 'right']
num_episodes = 500  # Number of learning episodes
max_steps = 100     # Maximum steps per episode
alpha = 0.1         # Learning rate
gamma = 0.9         # Discount factor
epsilon = 0.1       # Epsilon for exploration

# Initialize Q-table
Q = np.zeros((grid_size, grid_size, len(actions))) # 3 dimensions

# Define the rewards
reward_grid = np.zeros((grid_size, grid_size))
reward_grid[0, 4] = 1  # Goal state (top-right corner)

# Define the action directions (up, down, left, right)
action_effects = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# Epsilon-greedy policy
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Explore, randomly choose actions
    else:
        state_q_values = Q[state[0], state[1], :]
        return actions[np.argmax(state_q_values)]  # Exploit, pick specific action

# Take action and get the next state
def take_action(state, action):
    new_state = (state[0] + action_effects[action][0],
                 state[1] + action_effects[action][1])

    # Ensure the agent doesn't go out of bounds
    new_state = (max(0, min(grid_size - 1, new_state[0])),
                 max(0, min(grid_size - 1, new_state[1])))

    return new_state

# Q-Learning algorithm
for episode in range(num_episodes):
    # Start at a random state in the bottom-left corner (e.g., (4, 0))
    state = (4, 0)
    
    for step in range(max_steps):
        # Choose an action
        action = choose_action(state, epsilon)
        
        # Take the action and observe the new state and reward
        new_state = take_action(state, action)
        reward = reward_grid[new_state[0], new_state[1]]
        
        # Update Q-value using the Q-learning update rule
        action_index = actions.index(action)
        next_max_q = np.max(Q[new_state[0], new_state[1], :])
        
        Q[state[0], state[1], action_index] += alpha * (
            reward + gamma * next_max_q - Q[state[0], state[1], action_index])
        
        # Move to the next state
        state = new_state
        
        # End the episode if the agent reaches the goal
        if state == (0, 4):  # Goal state
            break

# Print the learned Q-table (rounded for readability)
print(np.round(Q, 2))


'''

NOTES: 

Grid Initialization: We create a 5x5 grid world and initialize a Q-table, Q, with zeros. 
Each state in the grid corresponds to an (x, y) position, and the Q-table stores the Q-values for each action 
(up, down, left, right) in every state.

Reward Grid: A separate reward_grid specifies the rewards. T
he agent receives +1 for reaching the goal (top-right corner) and 0 for all other movements.

Action Effects: The action_effects dictionary translates actions into state transitions 
(e.g., 'up' moves the agent one row up).

Epsilon-Greedy Policy: The function choose_action selects actions either randomly (exploration) 
or based on the highest Q-value for the current state (exploitation).

State Transitions: The take_action function ensures that actions result in 
valid new states without going out of bounds.

Q-Learning Update: After taking an action and observing the reward and the new state, 
the agent updates the Q-value using the Bellman equation. The Q-value for the state-action pair is adjusted toward the sum of the immediate reward and the discounted maximum future Q-value from the next state.

Training Loop: The agent is trained over multiple episodes to explore the grid and learn an optimal policy 
for reaching the goal.

'''


'''
HOW TO READ THE Q-TABLE:


Structure of the Q-Table
Rows represent the different states in the environment (in this case, positions on the 5x5 grid).
Columns represent the actions the agent can take from each state (up, down, left, right).
The values in the table are the Q-values, which tell us how "good" it is to take that action from that state.

In the above code, the Q-table is a 3D array of dimensions (5, 5, 4):

The first two dimensions correspond to the state (the position in the grid world).
The third dimension corresponds to the four actions (up, down, left, right).
For example, Q[2, 3, :] gives you the Q-values for all actions (up, down, left, right) 
when the agent is in state (2, 3) on the grid.

Suppose after training, the Q-table has values like this for a specific state:

Q[3, 2, :] = [0.1, -0.2, 0.0, 0.5]

The agent is at position (3, 2) on the grid.
The Q-values for the actions are:
Up: 0.1
Down: -0.2
Left: 0.0
Right: 0.5

The right action (0.5) has the highest Q-value, meaning the agent expects the 
best long-term reward by moving right from position (3, 2).
The Q-value for down is negative (-0.2), meaning that moving down is expected to 
lead to a bad outcome (perhaps moving farther from the goal).


To choose the best action from any given state, 
you would select the action with the highest Q-value for that state. For example, in state (3, 2):

best_action_index = np.argmax(Q[3, 2, :])
best_action = actions[best_action_index]

'''