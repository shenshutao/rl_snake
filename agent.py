import random
from collections import deque

import torch
import torch.optim as optim
# from agent_model import DQN  # 导入自定义的DQN模型
from ActorAndCriticModel import *
from torch.utils.tensorboard import SummaryWriter
import uuid


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
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=0.0001)
        self.criterion = torch.nn.MSELoss()  # 损失函数
        # self.writer = SummaryWriter('runs/'+uuid.uuid4().hex)
        # self.epoch = 0

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
                action_probs = self.actor(state)
            return action_probs.max(1)[1].item()  # 利用模型选择动作

    def replay(self, batch_size):
        # 检查记忆库中是否有足够的样本进行学习
        if len(self.memory.buffer) < batch_size:
            return  # 如果不够，直接返回，不进行训练

        # 从记忆库中随机抽取一批样本
        batch = self.memory.sample(batch_size)
        # 解压缩样本得到各个部分
        states, actions, rewards, next_states, dones = zip(*batch)

        # 将数据转换为适合模型输入的张量格式
        states = torch.FloatTensor(states)
        # next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(-1)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        # dones = torch.FloatTensor(dones).unsqueeze(-1)

        # 使用actor网络计算当前状态下，对数概率（log_probs），即执行的这个动作算出来的概率
        actions_probs = self.actor(states)
        action_probs = actions_probs.gather(1, actions)
        # log_probs = torch.log(action_probs)
        #
        # # 使用critic网络计算当前状态和下一个状态的价值估计
        # expected_values = self.critic(states)
        # with torch.no_grad():
        #     next_values = self.critic(next_states)

        # 计算目标价值，这里只考虑了即时奖励，而没有考虑未来奖励
        # 如果你想考虑未来的奖励，可以使用：targets = rewards + (0.99 * next_values * (1 - dones))
        targets = rewards

        # # 计算优势函数，即目标价值与当前价值的差异，优势函数（advantages）
        # advantages = targets - expected_values

        # # 根据优势函数和动作概率计算actor的损失，并执行梯度下降更新actor网络
        # # 白话：这个动作的概率 和 是否比平均价值高 相乘，如果有益，则最大化，如果有损，则最小化
        actor_loss = -(action_probs * targets.detach()).sum()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # # 根据优势函数计算critic的损失，并执行梯度下降更新critic网络
        # critic_loss = advantages.pow(2).mean()
        # self.optimizer_critic.zero_grad()
        # critic_loss.backward()
        # self.optimizer_critic.step()

        # self.epoch += 1
        #
        # if self.epoch%1000 == 0:
        #     print('epoch: ', self.epoch, 'actions_probs: ', actions_probs, 'action_probs: ', action_probs, ' rewards:', rewards, ' loss: ',  actor_loss)
        #
        # self.writer.add_scalar('Training loss', actor_loss, self.epoch)
        # for name, weight in self.actor.named_parameters():
        #     self.writer.add_histogram(name, weight, self.epoch)
        #     self.writer.add_histogram(f'{name}.grad', weight.grad, self.epoch)

        # 如果使用ε-贪婪策略进行探索，更新ε值
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return actor_loss

    def save(self, filename):
        # 保存模型参数
        torch.save(self.actor.state_dict(), 'actor_' + filename)
        torch.save(self.critic.state_dict(), 'critic_' + filename)

    def load(self, filename):
        # 加载模型参数
        self.actor.load_state_dict(torch.load('actor_' + filename))
        self.actor.eval()  # 设置为评估模式
