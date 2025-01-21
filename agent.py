import random
import torch
import torch.optim as optim
from collections import deque
from model import *


class Agent:
    def __init__(self, lr=0.0001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, batch_size=64):
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.model = DuelingDQN(11, 3)
        self.target_net = DuelingDQN(11, 3)
        self.model.load_state_dict(torch.load("./model/model.pth"))  # for continuing trained model
        self.model.train()  # for continuing trained model
        self.update_target_model()

        self.trainer = QTrainer(
            self.model,
            self.target_net,
            self.lr,
            self.gamma,
        )

        memory_limit = 64_000
        self.memory = deque(maxlen=memory_limit)
        self.transition = None
        self.n_games = 0
        self.record = 0
        # self.mean_score = 0

    def update_target_model(self):
        self.target_net.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def update_memory(self, state, action, reward, new_state, game_over):
        self.memory.append((state, action, reward, new_state, game_over))

    def train(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        if len(self.memory) < self.batch_size:
            return

        sample = random.sample(self.memory, self.batch_size)

        states, actions, rewards, new_states, game_overs = zip(*sample)
        self.trainer.train_step(states, actions, rewards, new_states, game_overs)
