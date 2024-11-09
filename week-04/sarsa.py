import numpy as np
import random

# Gridworld environment setup
class GridWorld:
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.state = (0, 0)  # Start at top-left corner
        self.goal = (grid_size - 1, grid_size - 1)  # Goal at bottom-right corner
        self.actions = ['up', 'down', 'left', 'right']
        
    def reset(self):
        self.state = (0, 0)
        return self.state
        
    def step(self, action):
        x, y = self.state
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < self.grid_size - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < self.grid_size - 1:
            y += 1
        
        self.state = (x, y)
        
        # If the agent reaches the goal, it's a success
        if self.state == self.goal:
            return self.state, 1  # Reward of 1 for reaching the goal
        else:
            return self.state, -0.1  # Small negative reward for each step taken

    def get_possible_actions(self):
        return self.actions

# SARSA agent
class SARSAAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Initialize Q-table with zeros (for each state-action pair)
        self.q_table = {}
        
    def get_q_value(self, state, action):
        # Initialize Q-value if not already present
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        return self.q_table[state].get(action, 0)
    
    def choose_action(self, state):
        # Epsilon-greedy strategy
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:
            # Exploit: Choose the action with the highest Q-value
            q_values = [self.get_q_value(state, action) for action in self.actions]
            max_q_value = max(q_values)
            best_actions = [self.actions[i] for i, q in enumerate(q_values) if q == max_q_value]
            return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state, next_action):
        current_q = self.get_q_value(state, action)
        next_q = self.get_q_value(next_state, next_action)
        # SARSA update rule
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

# Main Training Loop
def train_sarsa(episodes=1000, grid_size=4):
    env = GridWorld(grid_size)
    agent = SARSAAgent(env.get_possible_actions())
    
    for episode in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        
        done = False
        while not done:
            next_state, reward = env.step(action)
            next_action = agent.choose_action(next_state)
            
            agent.update_q_value(state, action, reward, next_state, next_action)
            
            state = next_state
            action = next_action
            
            # If goal is reached, end episode
            if state == env.goal:
                done = True
                
        # Optionally print Q-table after every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}")
            print(agent.q_table)
    
    return agent

# Run the training
agent = train_sarsa(episodes=1000, grid_size=4)

# Display learned Q-values
print("Learned Q-values:")
print(agent.q_table)

'''

Explanation of the Code:

GridWorld Environment:

The agent starts in the top-left corner of a 4x4 grid.
The goal is to reach the bottom-right corner.
The agent can take four possible actions: 'up', 'down', 'left', or 'right'.
The reward is -0.1 for every step, and +1 when the agent reaches the goal.

SARSA Agent:

The agent stores Q-values in a table (q_table), which is initialized to zero 
for all state-action pairs.
The choose_action method uses an epsilon-greedy policy to either explore new actions 
randomly or exploit the learned Q-values.
The update_q_value method applies the SARSA update rule to update the Q-values 
based on the current state, action, reward, next state, and next action.

Training Loop:

The agent explores the environment for a specified number of episodes (1000 in this case).
For each episode, the agent interacts with the environment and updates its 
Q-table using SARSA.

Q-table:

After training, the agent's Q-table will show the learned values for each state-action pair.
Over time, the agent will learn which actions lead to the goal and improve its policy.

How SARSA Works in This Example:
At each step, the agent takes an action, receives a reward, and transitions to a new state.
The agent uses the SARSA update rule to adjust the Q-value of the state-action pair based 
on the reward and the value of the next state-action pair.
Since SARSA is an on-policy algorithm, the update uses the action selected by the 
agent (according to its current policy), not necessarily the optimal action.




'''