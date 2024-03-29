import pygame
import numpy as np
import random


class SnakeEnv:
    def __init__(self, mode='TEST', grid_size=20, block_size=20):
        """初始化游戏环境"""
        self.grid_size = grid_size  # 网格大小
        self.block_size = block_size  # 每个网格块的大小
        self.screen_size = grid_size * block_size  # 屏幕尺寸

        # 如果是测试模式，则初始化Pygame窗口和相关设置
        if mode == 'TEST':
            pygame.init()
            self.dis = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption('Snake Game')  # 设置窗口标题
            self.clock = pygame.time.Clock()  # 控制游戏速度
            self.font_style = pygame.font.SysFont(None, 35)  # 设置字体

        self.reset()  # 重置游戏到初始状态

    def get_env_info(self):
        """获取环境信息，包括状态大小和动作大小"""
        state_size = 6  # 状态信息大小：蛇头位置和食物位置
        action_size = 4  # 动作大小：上下左右
        return state_size, action_size

    def reset(self):
        """重置游戏环境到初始状态"""
        self.snake_pos = [[self.grid_size // 2, self.grid_size // 2]]  # 蛇的初始位置，开始时位于中心
        self.food_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]  # 随机生成食物位置
        self.direction = 'STOP'  # 初始方向为停止
        self.score = 0  # 分数重置为0
        self.done = False  # 游戏状态标志，False表示游戏未结束
        return self._get_state()  # 返回初始状态

    def _get_state(self):
        """获取当前环境状态"""
        x, y = self.snake_pos[0]  # 蛇头位置
        fx, fy = self.food_pos  # 食物位置

        # 状态数组，包含食物相对蛇头的位置以及蛇头的坐标
        state_arr = [fx < x,  # 食物在蛇头左侧
                     fx > x,  # 食物在蛇头右侧
                     fy > y,  # 食物在蛇头上方
                     fy < y,  # 食物在蛇头下方
                     x,  # 蛇头X坐标
                     y]  # 蛇头Y坐标

        state = np.array(state_arr, dtype=int)
        return state

    def step(self, action):
        """根据动作更新环境状态，并返回新的状态、奖励和游戏是否结束"""
        # 定义动作对应的方向
        directions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        dx, dy = 0, 0
        if directions[action] == 'LEFT':
            dx, dy = -1, 0
        elif directions[action] == 'RIGHT':
            dx, dy = 1, 0
        elif directions[action] == 'UP':
            dx, dy = 0, 1
        elif directions[action] == 'DOWN':
            dx, dy = 0, -1

        # 更新蛇头位置
        previous_head = [self.snake_pos[0][0], self.snake_pos[0][1]]
        head = [self.snake_pos[0][0] + dx, self.snake_pos[0][1] + dy]
        self.snake_pos.insert(0, head)

        # 检查是否吃到食物
        if head == self.food_pos:
            self.score += 1  # 吃到食物分数加1
            self.food_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]  # 生成新的食物位置
        else:
            self.snake_pos.pop()  # 没吃到食物，蛇尾部去掉一个单位长度

        if abs(previous_head[0] - self.food_pos[0]) > abs(head[0] - self.food_pos[0]) or abs(previous_head[0] - self.food_pos[0]) > abs(head[0] - self.food_pos[0]):
            self.score += 1
        else:
            self.score -= 1

        # 这里可以添加检查游戏是否结束的逻辑，此处为了简化暂时不写，后续版本再补

        return self._get_state(), self.score, self.done

    def render(self):
        """渲染游戏画面"""
        self.dis.fill((0, 0, 0))  # 用黑色填充屏幕

        # 绘制蛇头，使用蓝色
        pygame.draw.rect(self.dis, (0, 0, 255),
                         [self.snake_pos[0][1] * self.block_size, self.snake_pos[0][0] * self.block_size,
                          self.block_size, self.block_size])

        # 绘制蛇身，使用绿色
        for part in self.snake_pos[1:]:
            pygame.draw.rect(self.dis, (0, 255, 0),
                             [part[1] * self.block_size, part[0] * self.block_size, self.block_size, self.block_size])

        # 绘制食物，使用红色
        pygame.draw.rect(self.dis, (255, 0, 0),
                         [self.food_pos[1] * self.block_size, self.food_pos[0] * self.block_size, self.block_size,
                          self.block_size])

        pygame.display.update()  # 更新屏幕显示
        self.clock.tick(10)  # 控制游戏速度

    def close(self):
        """关闭游戏窗口"""
        pygame.quit()


if __name__ == "__main__":
    # 示例使用
    env = SnakeEnv()
    state = env.reset()

    for _ in range(10000):  # 设定运行步数
        action = random.randint(0, 3)  # 随机选择动作
        state, score, done = env.step(action)
        env.render()  # 渲染游戏画面
        if done:
            break

    env.close()  # 关闭游戏