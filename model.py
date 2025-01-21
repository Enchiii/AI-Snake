import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class DuelingDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, 128)

        self.dropout = nn.Dropout(p=0.3)

        self.value_linear = nn.Linear(128, 64)
        self.value_output = nn.Linear(64, 1)

        self.advantage_linear = nn.Linear(128, 64)
        self.advantage_output = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)

        value = F.relu(self.value_linear(x))
        value = self.value_output(value)

        advantage = F.relu(self.advantage_linear(x))
        advantage = self.advantage_output(advantage)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        preds = self.model(states)
        targets = preds.clone()
        for idx in range(len(dones)):
            q_new = rewards[idx]
            if not dones[idx]:
                q_new = rewards[idx] + self.gamma * torch.max(self.target_model(next_states[idx].unsqueeze(0)))

            targets[idx][torch.argmax(actions[idx]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(preds, targets)
        loss.backward()
        self.optimizer.step()
