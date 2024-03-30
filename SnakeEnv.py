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
        state_size = 4  # 状态信息大小：食物位置和蛇头位置
        action_size = 4  # 动作大小：左右上下
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

        # 状态数组
        state_arr = [
                    fx, # 食物x坐标
                    fy, # 食物y坐标
                    x,  # 蛇头X坐标
                    y]  # 蛇头Y坐标

        state = np.array(state_arr, dtype=int)
        return state
        # """
        #     创建一个游戏状态二维数组。
        #
        #     参数:
        #     - grid_size: 游戏网格的大小，假设是一个正方形。
        #     - snake_positions: 蛇身体的位置列表，包括蛇头。蛇头在列表的第一个位置。
        #     - food_position: 食物的位置。
        #
        #     返回:
        #     - 一个二维数组，表示游戏的当前状态。
        #     """
        # # 初始化一个grid_size x grid_size的二维数组，初始值为0
        # game_state = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        #
        # # 标记食物的位置
        # game_state[self.food_pos[0]][self.food_pos[1]] = 3
        #
        # # 标记蛇头的位置
        # snake_head = self.snake_pos[0]
        # game_state[snake_head[0]][snake_head[1]] = 1
        #
        # # 标记蛇身的位置
        # for pos in self.snake_pos[1:]:
        #     game_state[pos[0]][pos[1]] = 2
        #
        # return np.array(game_state).flatten()

    def step(self, action):
        """根据动作更新游戏状态，并检查游戏是否结束。"""
        directions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        dx, dy = {'LEFT': (-1, 0), 'RIGHT': (1, 0), 'UP': (0, 1), 'DOWN': (0, -1)}[directions[action]]

        previous_head = [self.snake_pos[0][0], self.snake_pos[0][1]]
        new_head = [self.snake_pos[0][0] + dx, self.snake_pos[0][1] + dy]

        # 检查蛇头是否撞到边界
        if new_head[0] < 0 or new_head[0] >= self.grid_size or new_head[1] < 0 or new_head[1] >= self.grid_size:
            self.score = -10
            self.done = True

        # # 检查蛇头是否撞到自己的身体
        # if new_head in self.snake_pos:
        #     self.score = -10
        #     self.done = True

        # 如果游戏未结束，更新蛇的位置
        if not self.done:
            self.snake_pos.insert(0, new_head)
            if new_head == self.food_pos:  # 吃到食物
                self.score += 10
                self.food_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            else:
                self.snake_pos.pop()  # 移动蛇，未吃到食物则移除尾部

            # 引导，向食物靠近则+0.1，远离则-0.1
            if (abs(previous_head[0] - self.food_pos[0]) > abs(new_head[0] - self.food_pos[0])) or (abs(previous_head[1] - self.food_pos[1]) > abs(new_head[1] - self.food_pos[1])):
                self.score += 1
            else:
                self.score -= 1

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