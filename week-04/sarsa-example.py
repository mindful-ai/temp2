import numpy as np
import random

# Define the grid dimensions
grid_size = 4
start = (0, 0)
target = (2, 2)

# Parameters for SARSA
alpha = 0.1    # Learning rate
gamma = 0.9    # Discount factor
epsilon = 0.1  # Exploration rate

# Actions: up, down, left, right
actions = ['up', 'down', 'left', 'right']

# Initialize the Q-table with zeros for each state-action pair
Q = {}
for row in range(grid_size):
    for col in range(grid_size):
        Q[(row, col)] = {action: 0 for action in actions}

# Helper function to choose an action based on epsilon-greedy policy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Explore: random action
    else:
        # Exploit: choose the action with max Q-value for current state
        return max(Q[state], key=Q[state].get)

# Function to take a step in the environment
def step(state, action):
    row, col = state
    if action == 'up':
        next_state = (max(row - 1, 0), col)
    elif action == 'down':
        next_state = (min(row + 1, grid_size - 1), col)
    elif action == 'left':
        next_state = (row, max(col - 1, 0))
    elif action == 'right':
        next_state = (row, min(col + 1, grid_size - 1))
    else:
        next_state = state
    
    # Reward logic
    if next_state == target:
        return next_state, 1  # Positive reward for reaching the target
    else:
        return next_state, -0.1  # Small penalty for each step

# SARSA algorithm
num_episodes = 1000
for episode in range(num_episodes):
    state = start
    action = choose_action(state)
    
    while state != target:
        # Take action, observe reward and next state
        next_state, reward = step(state, action)
        next_action = choose_action(next_state)
        
        # SARSA update rule
        Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
        
        # Move to the next state-action pair
        state = next_state
        action = next_action

# Display the learned Q-values
print("Learned Q-values:")
for row in range(grid_size):
    for col in range(grid_size):
        print(f"State ({row}, {col}): {Q[(row, col)]}")

# Display the optimal policy
print("\nOptimal Policy:")
for row in range(grid_size):
    for col in range(grid_size):
        if (row, col) == target:
            print(" T ", end="\t")  # Target
        else:
            best_action = max(Q[(row, col)], key=Q[(row, col)].get)
            print(f"{best_action[:2]}", end="\t")
    print()
