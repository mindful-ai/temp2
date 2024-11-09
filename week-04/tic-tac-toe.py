
import numpy as np
import random
import pickle

class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9)  # Initialize the board as a 1D array (9 positions)
        self.done = False
        self.winner = None
    
    def reset(self):
        self.board = np.zeros(9)
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return tuple(self.board)  # Returns the current state of the board as a tuple

    def available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def check_winner(self):
        winning_positions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                             (0, 3, 6), (1, 4, 7), (2, 5, 8),
                             (0, 4, 8), (2, 4, 6)]
        for p1, p2, p3 in winning_positions:
            if self.board[p1] == self.board[p2] == self.board[p3] and self.board[p1] != 0:
                self.winner = self.board[p1]
                self.done = True
                return True
        if not self.available_actions():  # Draw if no available actions
            self.winner = 0
            self.done = True
            return True
        return False

    def step(self, action, player):
        if self.board[action] == 0 and not self.done:
            self.board[action] = player
            self.check_winner()
            return self.get_state(), 1 if self.winner == player else 0, self.done
        else:
            return self.get_state(), -1, self.done  # Invalid move penalty


class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration factor

    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < self.epsilon:  # Explore
            return random.choice(available_actions)
        else:  # Exploit
            q_values = [self.q_table.get((state, a), 0) for a in available_actions]
            max_q = max(q_values)
            return available_actions[q_values.index(max_q)]

    def update_q_table(self, state, action, reward, next_state, next_available_actions):
        old_q_value = self.q_table.get((state, action), 0)
        next_max_q = max([self.q_table.get((next_state, a), 0) for a in next_available_actions], default=0)
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * next_max_q - old_q_value)
        self.q_table[(state, action)] = new_q_value


def train_q_learning_agent(episodes=10000):
    agent = QLearningAgent()
    env = TicTacToe()

    for _ in range(episodes):
        state = env.reset()
        done = False
        player = 1

        while not done:
            available_actions = env.available_actions()
            action = agent.choose_action(state, available_actions)
            next_state, reward, done = env.step(action, player)

            if done and reward == 0:  # Draw
                reward = 0.5

            next_available_actions = env.available_actions() if not done else []
            agent.update_q_table(state, action, reward, next_state, next_available_actions)

            state = next_state
            player = -player  # Switch player

    return agent


def play_game(agent):
    env = TicTacToe()
    state = env.reset()
    done = False
    player = 1  # Human starts

    while not done:
        if player == 1:
            print("Board State:")
            print(env.board.reshape(3, 3))
            available_actions = env.available_actions()
            action = int(input(f"Choose your action (0-8): {available_actions} "))
        else:
            available_actions = env.available_actions()
            action = agent.choose_action(state, available_actions)
            print(f"AI chose action: {action}")

        next_state, reward, done = env.step(action, player)
        state = next_state
        player = -player

    print("Final Board State:")
    print(env.board.reshape(3, 3))

    if env.winner == 1:
        print("You won!")
    elif env.winner == -1:
        print("AI won!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    # Train the agent
    agent = train_q_learning_agent(episodes=10000)

    # Save the agent
    with open("tic_tac_toe_agent.pkl", "wb") as f:
        pickle.dump(agent, f)

    # Load and play the game
    with open("tic_tac_toe_agent.pkl", "rb") as f:
        trained_agent = pickle.load(f)

    play_game(trained_agent)

'''

Explanation:

TicTacToe class: Handles the game logic, board state, available actions, and checks for a winner.

QLearningAgent class: Implements the Q-learning algorithm. It stores the Q-table, selects actions 
based on the epsilon-greedy policy, and updates the Q-values.

train_q_learning_agent function: Trains the agent over a series of episodes (10,000 in this case). 
The agent plays against itself, and Q-values are updated after every move.

play_game function: Allows a human player to play against the trained Q-learning agent. 
The board state is printed and the human can input moves.

How It Works:

The agent learns by playing against itself using Q-learning.
The Q-table stores the expected reward for each state-action pair.
Once trained, the agent can make optimal moves in new games.

Running the Code:

Train the agent with train_q_learning_agent().
You can save and load the trained agent using the pickle module.
The play_game() function lets a human player compete against the trained AI.

'''