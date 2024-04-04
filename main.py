import time

import matplotlib.pyplot as plt

from SnakeEnv import SnakeEnv
from agents.DQNAgent import Agent

# 初始化蛇游戏环境和智能体
env = SnakeEnv(mode='TRAIN')
input_channels, state_size, action_size = env.get_env_info()  # 状态和动作的维度
agent = Agent(input_channels, state_size, action_size)

# 训练参数设置
episodes = 10000000  # 总训练回合数
batch_size = 512  # 每轮步数
episode_rewards = []  # 用于存储每个episode的总奖励

agent.load("agent_model.pth")  # 加载预训练模型，电脑慢的可以重复训练一个模型
print('Start training...')
start_time = time.time()

# 训练循环
for e in range(episodes):
    state, expect_action = env.reset()  # 重置环境获取初始状态
    current_score = 0  # 该回合总得分初始化为0
    previous_score = 0  # 初始化前一得分

    for s in range(batch_size):
        action = agent.act(state)  # 根据当前状态选择动作
        # print(action, end=" ")
        next_state, expect_action, current_score, done = env.step(action)  # 执行动作，获得反馈
        # env.render()  # 更新当前环境的图像
        reward = current_score - previous_score  # 计算当前步骤的即时奖励
        agent.remember(state, action, reward, next_state, done, expect_action)  # 保存经验
        state = next_state
        previous_score = current_score
        if done:
            # print('蛇死了')
            break

    episode_rewards.append(current_score)  # 记录本回合总得分
    loss = agent.replay()  # 经验回放学习
    # print()
    # print(f'Start replay for episode {e}, step {s}, total_score is {current_score}, loss is {loss}')

    # 每1000个episode保存一次模型，并打印信息
    if e % 1000 == 0:
        print(f'Start replay for episode {e}, step {s}, total_score is {current_score}, loss is {loss}')
        print(agent.memory.sample(1)[0])  # 打印一个样本经验，以观察
        agent.save('agent_model.pth')  # 保存模型

end_time = time.time()
print(f'Time cost: {end_time - start_time}')

# 绘制训练过程中的奖励趋势
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Score')
plt.show()

# 测试部分
print('Start testing')
agent.load("agent_model.pth")  # 加载模型
env = SnakeEnv(mode='TEST')  # 设置环境为测试模式

state, expect_action = env.reset()  # 重置环境获取初始状态
for s in range(1000):  # 这里假设最多执行1000步
    agent.epsilon = 0  # 设置epsilon=0，完全依赖模型进行决策
    action = agent.act(state)  # 选择动作
    next_state, expect_action, reward, done = env.step(action)
    img = env.render()  # 获取当前环境的图像

    state = next_state
    if done:
        break

# 注意：你可以在循环中保存每一步的图像，后续可以将它们组合成视频。
