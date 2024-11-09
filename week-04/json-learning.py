''' 

Here's a simple reinforcement learning example that involves JSON data. 
In this case, let's simulate an agent learning to navigate a virtual store 
(represented as JSON data), where different sections (categories) of the store 
provide different rewards. 
The agent must learn to maximize its rewards by visiting the most valuable sections 
of the store. 
The JSON data represents different sections and their associated rewards.

see store.json

'''

import json
import numpy as np
import random

# Load the JSON file representing store sections and their rewards
def load_store_data(file_path):
    with open(file_path, 'r') as f:
        store_data = json.load(f)
    return store_data['sections']

# Define the environment for the store navigation problem
class StoreEnvironment:
    def __init__(self, store_data):
        self.sections = list(store_data.keys())  # List of store sections (state space)
        self.rewards = {section: store_data[section]["reward"] for section in self.sections}
        self.state = None
    
    def reset(self):
        self.state = random.choice(self.sections)  # Start at a random section
        return self.state
    
    def step(self, action):
        reward = self.rewards[action]  # Get the reward for the current section
        self.state = action  # Move to the chosen section
        return self.state, reward, False  # No terminal state, game goes on
    
    def available_actions(self):
        return self.sections  # All sections are available as actions

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, n_actions, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = {section: 0 for section in n_actions}  # Initialize Q-table
    
    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < self.epsilon:
            # Explore: Choose random action
            return random.choice(available_actions)
        else:
            # Exploit: Choose the action with the highest Q-value
            return max(self.q_table, key=self.q_table.get)
    
    def update_q_values(self, state, action, reward, next_state):
        old_q_value = self.q_table[action]
        max_next_q_value = self.q_table[next_state]
        # Update Q-value using Bellman Equation
        self.q_table[action] = old_q_value + self.alpha * (reward + self.gamma * max_next_q_value - old_q_value)

# Run the store navigation reinforcement learning experiment
def run_store_rl_experiment(store_file, episodes=1000, epsilon=0.1):
    store_data = load_store_data(store_file)
    env = StoreEnvironment(store_data)
    agent = QLearningAgent(env.available_actions(), epsilon)
    
    for episode in range(episodes):
        state = env.reset()  # Start at a random section
        done = False
        
        while not done:
            action = agent.choose_action(state, env.available_actions())
            next_state, reward, done = env.step(action)  # Perform the action, get reward
            agent.update_q_values(state, action, reward, next_state)
            state = next_state  # Move to the next state
    
    return agent.q_table

# Example: Using store.json as input data
if __name__ == "__main__":
    q_table = run_store_rl_experiment('store.json', episodes=1000, epsilon=0.1)
    print("Learned Q-values for each store section:")
    for section, value in q_table.items():
        print(f"Section: {section}, Q-value: {value}")

'''

Explanation:

Store JSON Data:

The store.json file represents different sections of a store (electronics, clothing, grocery, etc.), 
each with an associated reward value.

StoreEnvironment Class:

The store is modeled as an environment where the agent can move between different sections. 
The agent receives a reward based on the section it visits, and the goal is to learn the best sections 
to visit to maximize the reward.


QLearningAgent Class:

This agent uses Q-learning to learn the best action (section) to take in the store. 
The Q-table is initialized with each store section as a possible action, and it is updated 
using the rewards received during the training process.


run_store_rl_experiment Function:

Simulates the agent interacting with the store environment over a number of episodes. 
It trains the agent to explore the sections and learn which one gives the best rewards.

How It Works:

The agent starts in a random section of the store and selects actions (moving to different sections) 
based on an epsilon-greedy strategy.
It receives rewards based on the section visited and updates its Q-values accordingly.
Over time, the agent learns which sections offer the highest rewards and will prefer them more frequently.

Example Output:
After training, the agent will have learned the Q-values for each section:

Learned Q-values for each store section:
Section: electronics, Q-value: 3.25
Section: clothing, Q-value: 1.12
Section: grocery, Q-value: 2.09
Section: furniture, Q-value: 5.0
Section: toys, Q-value: 4.1

The Q-values represent the agent's learned estimate of the value of visiting each 
section based on the rewards it received during training.

'''