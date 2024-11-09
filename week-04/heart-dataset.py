'''

Here’s a simple reinforcement learning (RL) example using the UCI Heart Disease Dataset. 
We will simulate an environment where the RL agent tries to predict whether a patient has 
heart disease based on the dataset and gets rewards based on its predictions. The goal is 
to train the agent to make the right decisions (classifications) by learning from rewards.


Dataset: The Heart Disease dataset in CSV format will be used.
Environment: The RL environment will give a reward based on the correctness of the prediction (disease or no disease).
RL Algorithm: We’ll use a basic Q-learning approach.


We’ll treat this as a classification problem where the RL agent receives a reward for correct predictions. 
This is a basic Q-learning setup that interacts with the environment (dataset) and updates Q-values based on 
whether the prediction was correct or not.

'''

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('heart.csv')

# Split data into features (X) and target (y)
X = data.drop(columns=['target'])
y = data['target']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the environment for Q-learning
class HeartDiseaseEnv:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.state = 0  # Start at the first sample
    
    def reset(self):
        self.state = 0
        return self.X[self.state]
    
    def step(self, action):
        correct = (action == self.y[self.state])  # Action is either 0 (no disease) or 1 (disease)
        reward = 1 if correct else -1  # Reward for correct prediction
        self.state += 1
        done = (self.state >= self.n_samples)
        next_state = self.X[self.state] if not done else None
        return next_state, reward, done

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, n_actions, state_size, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = np.zeros((state_size, n_actions))  # Initialize Q-table
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])  # Explore: Choose randomly (0 = no disease, 1 = disease)
        else:
            return np.argmax(self.q_table[state])  # Exploit: Choose the action with the highest Q-value
    
    def update_q_values(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state]) if next_state is not None else 0
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action] if next_state is not None else reward
        self.q_table[state, action] += self.alpha * (td_target - self.q_table[state, action])

# Run Q-learning on the dataset
def run_ql_heart_disease(X_train, y_train, n_episodes=1000, epsilon=0.1):
    env = HeartDiseaseEnv(X_train, y_train)
    agent = QLearningAgent(n_actions=2, state_size=X_train.shape[1], epsilon=epsilon)
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_values(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        if episode % 100 == 0:
            print(f'Episode {episode}, Total Reward: {total_reward}')
    
    return agent.q_table

# Example of running Q-learning
q_table = run_ql_heart_disease(X_train, y_train, n_episodes=1000, epsilon=0.1)
print("Learned Q-table:")
print(q_table)


'''

Dataset: We load and preprocess the heart disease dataset (normalizing features and splitting into training and test sets).

HeartDiseaseEnv: The environment class simulates the RL environment where the agent interacts with the dataset. 
The agent makes predictions (action 0 or 1) for each sample, and the environment returns a reward based on whether 
the prediction was correct.

QLearningAgent: The Q-learning agent learns to predict heart disease by updating its Q-values based on the 
rewards from the environment.

Q-Table: The agent builds a Q-table, where it learns the value of each state-action pair. 
In this case, the states are feature vectors (representing a patient’s data), and the actions are the predictions (whether the patient has heart disease or not).

Training Loop: The agent is trained for a certain number of episodes, where it interacts 
with the environment (the heart disease dataset) and updates its Q-table accordingly.


'''