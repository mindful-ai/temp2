'''
In this 4x4 grid:

Each cell is a "state."
The agent (let’s call it “Robo”) can move up, down, left, or right.
The agent’s goal is to reach a target cell where it receives a reward of +1.
Each move costs -0.1 to encourage the shortest path.
The agent learns which moves maximize its cumulative rewards over episodes.

'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque

# Hyperparameters
gamma = 0.9              # Discount factor
epsilon = 1.0            # Exploration rate
epsilon_decay = 0.995    # Epsilon decay after each episode
min_epsilon = 0.01       # Minimum exploration rate
alpha = 0.001            # Learning rate
batch_size = 32          # Batch size for training
memory_size = 1000       # Replay memory size
num_episodes = 500       # Number of episodes

# Gridworld environment setup
grid_size = 4
start = (0, 0)
target = (2, 2)

# Possible actions
actions = ['up', 'down', 'left', 'right']
action_to_index = {action: i for i, action in enumerate(actions)}

# Initialize Replay Memory
replay_memory = deque(maxlen=memory_size)

# Define Q-Network using TensorFlow
def create_q_network():
    model = tf.keras.Sequential([
        layers.Input(shape=(grid_size * grid_size,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(actions), activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')
    return model

# Initialize the Q-Network
q_network = create_q_network()

# Function to convert a state (row, col) into a one-hot encoded vector
def one_hot_state(state):
    vector = np.zeros(grid_size * grid_size)
    index = state[0] * grid_size + state[1]
    vector[index] = 1
    return np.array(vector, dtype=np.float32)

# Function to select an action based on epsilon-greedy policy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Explore
    else:
        state_tensor = np.expand_dims(one_hot_state(state), axis=0)
        q_values = q_network.predict(state_tensor, verbose=0)
        best_action_index = np.argmax(q_values[0])
        return actions[best_action_index]  # Exploit

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
        return next_state, 1, True  # Positive reward and end episode
    else:
        return next_state, -0.1, False  # Small penalty for each step

# Training Loop
for episode in range(num_episodes):
    state = start
    done = False
    global epsilon

    while not done:
        # Select action and observe next state and reward
        action = choose_action(state)
        next_state, reward, done = step(state, action)

        # Store experience in replay memory
        replay_memory.append((state, action, reward, next_state, done))
        
        # Sample a batch of experiences from memory
        if len(replay_memory) >= batch_size:
            batch = random.sample(replay_memory, batch_size)
            states, actions_batch, rewards, next_states, dones = zip(*batch)

            # Prepare input arrays
            states = np.array([one_hot_state(s) for s in states])
            next_states = np.array([one_hot_state(ns) for ns in next_states])
            rewards = np.array(rewards)
            dones = np.array(dones, dtype=np.float32)

            # Q(s, a) predictions
            q_values = q_network.predict(states, verbose=0)
            next_q_values = q_network.predict(next_states, verbose=0)

            # Update Q-value targets
            target_q_values = q_values.copy()
            for i in range(batch_size):
                action_index = action_to_index[actions_batch[i]]
                if dones[i]:
                    target_q_values[i][action_index] = rewards[i]  # No future reward if done
                else:
                    target_q_values[i][action_index] = rewards[i] + gamma * np.max(next_q_values[i])

            # Train on the batch of state-action pairs
            q_network.train_on_batch(states, target_q_values)

        # Move to the next state
        state = next_state

    # Decay epsilon after each episode
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

print("Training completed!")

'''
Q-Network Definition:

The create_q_network() function creates a neural network using TensorFlow’s Keras 
API.
The network consists of two hidden layers with 64 neurons each, 
using ReLU activation. The output layer has one output for each action 
(up, down, left, right).

Action Selection (Epsilon-Greedy Policy):

choose_action() decides whether to explore or exploit based on epsilon.
If exploring, it randomly selects an action. Otherwise, it picks the action 
with the highest predicted Q-value.

Experience Replay Memory:

Experiences are stored in a deque (replay memory) and are randomly 
sampled to break correlation in training data and stabilize training.

Training the Network:

Each episode updates the Q-values for each state-action pair 
using a randomly sampled batch from the replay memory.
Q-values are updated using the Bellman equation.
The Q-network is trained in batches, where the target 
Q-values are calculated as the observed reward plus the discounted maximum Q-value from the next state.
Epsilon Decay:

The epsilon parameter decays over time, so the agent shifts 
from exploring to exploiting as training progresses.

'''