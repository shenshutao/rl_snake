import random
from collections import deque

import torch
import torch.optim as optim
from agent_model import DQN  # 导入自定义的DQN模型

class ReplayBuffer:
    """记忆回放缓存，用于存储和随机抽样过去的经验，以进行训练"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 使用双端队列作为经验缓存

    def add(self, state, action, reward, next_state, done):
        # 向缓存中添加一个经验元组
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 随机抽样一批经验
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

class Agent:
    """基于深度Q网络(DQN)的强化学习智能体"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # 状态空间的大小
        self.action_size = action_size  # 动作空间的大小
        self.memory = ReplayBuffer(500)  # 初始化记忆回放缓存
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减因子
        self.model = DQN(state_size, action_size)  # DQN模型
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # 优化器
        self.criterion = torch.nn.MSELoss()  # 损失函数

    def remember(self, state, action, reward, next_state, done):
        # 存储经验
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state):
        # 选择一个动作
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)  # 探索新动作
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.model(state)
            return action_values.max(1)[1].item()  # 利用模型选择动作

    def replay(self, batch_size):
        # 从记忆回放中抽样，并使用DQN进行训练
        if len(self.memory.buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = zip(*self.memory.sample(batch_size))
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(-1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        Q_expected = self.model(states).gather(1, actions)
        targets = rewards.unsqueeze(1)

        self.optimizer.zero_grad()
        loss = self.criterion(Q_expected, targets)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # 更新探索率

    def save(self, filename):
        # 保存模型参数
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        # 加载模型参数
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()  # 设置为评估模式