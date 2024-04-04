import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# DQN model
class DQN(nn.Module):
    def __init__(self, input_channels, action_size):
        super(DQN, self).__init__()
        # CNN层
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # 全联接层
        self.fc = nn.Sequential(
            nn.Linear(32 * 10 * 10, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)  # 展平
        x = self.fc(x)
        return x


class ReplayBuffer:
    """记忆回放缓存，用于存储和随机抽样过去的经验，以进行训练"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 使用双端队列作为经验缓存

    def add(self, state, action, reward, next_state, done, expect_actions):
        # 向缓存中添加一个经验元组
        self.buffer.append((state, action, reward, next_state, done, expect_actions))

    def sample(self, batch_size):
        # 随机抽样一批经验
        return random.sample(self.buffer, min(len(self.buffer), batch_size))


class Agent:
    def __init__(self, input_channels, state_size, action_size, batch_size=128):
        self.state_size = state_size  # 状态空间的大小
        self.action_size = action_size  # 动作空间的大小
        self.memory = ReplayBuffer(500)  # 初始化记忆回放缓存
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.999  # 探索率衰减因子
        self.batch_size = batch_size

        self.model = DQN(input_channels, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def remember(self, state, action, reward, next_state, done, expect_action):
        self.memory.add(state, action, reward, next_state, done, expect_action)

    def act(self, state):
        # 选择一个动作
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)  # 探索新动作
        else:
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                action_values = self.model(state)
        return np.argmax(action_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory.buffer) < self.batch_size:
            return  # 如果不够，直接返回，不进行训练

        # 从记忆库中随机抽取一批样本
        batch = self.memory.sample(self.batch_size)
        # 解压缩样本得到各个部分
        states, actions, rewards, next_states, dones, expect_actions = zip(*batch)

        states = torch.stack([torch.Tensor(state).unsqueeze(0) for state in states])
        next_states = torch.stack([torch.Tensor(next_state).unsqueeze(0) for next_state in next_states])
        actions = torch.LongTensor(actions).unsqueeze(-1)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        dones = torch.FloatTensor(dones).unsqueeze(-1)

        Q_targets_next = self.model(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))  # 实际使用该action的整体收益 = 当前收益 + 后续收益
        Q_expected = self.model(states).gather(1, actions)  # 在状态state下，使用action，模型预测的收益Q_expected
        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, filename):
        # 保存模型参数
        torch.save(self.model.state_dict(), 'DQN_' + filename)

    def load(self, filename):
        # 加载模型参数
        self.model.load_state_dict(torch.load('DQN_' + filename))
        self.model.eval()  # 设置为评估模式
