import torch
import torch.nn as nn
import numpy as np


# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # 定义全连接层
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, x):
        # 前向传播定义，使用ReLU激活函数
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


if __name__ == "__main__":
    state = [0, 0, 0, 0, 11, 9]
    state = torch.FloatTensor(state).unsqueeze(0)  # 将状态转换为适合模型的形式

    # 创建DQN模型实例
    model = DQN(6, 4)

    with torch.no_grad():  # 禁用梯度计算，因为这里只是进行前向传播
        output = model(state)

        # 使用Softmax获取动作的概率分布
        softmax_output = torch.softmax(output, dim=1)
        print("Softmax output:", softmax_output)

        # 获取概率最大的动作
        _, predicted_class = torch.max(output, 1)
        predicted_class_int = predicted_class.item()
        print("Predicted action:", predicted_class_int)

        # 使用Sigmoid函数处理输出，然后根据阈值决定动作（可选，根据需求选择）
        probabilities = torch.sigmoid(output)
        print("Sigmoid probabilities:", probabilities)
        threshold = 0.5
        predictions = (probabilities > threshold).int()  # 大于阈值的为1，否则为0
        print("Action predictions based on threshold:", predictions)
