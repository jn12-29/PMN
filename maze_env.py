#!/usr/bin/env python3
"""
基于本体感觉的迷宫导航 Gymnasium 环境
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import labmaze
from utils import (
    OBS_DICT,
    Direction,
    DEFAULT_MAZE_LAYOUT,
    find_positions,
    encode_cell,
    render_frame,
)


class MazeNavigationEnv(gym.Env):
    """
    基于本体感觉的迷宫导航环境

    观测空间：
        - pos_aware=True: [current_row, current_col, target_row, target_col, up_cell, down_cell, left_cell, right_cell]
        - pos_aware=False: [up_cell, down_cell, left_cell, right_cell]
    动作空间：4个离散动作（上、下、左、右）
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
        self,
        render_mode="rgb_array",
        maze_layout=None,
        num_spawns=1,
        num_objects=1,
        pos_aware=False,
    ):
        """
        初始化迷宫导航环境

        Args:
            render_mode: 渲染模式（"human" 或 "rgb_array"）
            maze_layout: 自定义迷宫布局（字符串），如果为 None 则使用默认布局
            num_spawns: 生成点数量
            num_objects: 目标对象数量
            pos_aware: 是否包含位置信息（当前位置和目标位置），默认为 False
        """
        super().__init__()

        self.pos_aware = pos_aware
        # 设置渲染模式
        self.render_mode = render_mode

        # 创建迷宫
        if maze_layout is None:
            maze_layout = DEFAULT_MAZE_LAYOUT

        self.maze = labmaze.FixedMazeWithRandomGoals(
            entity_layer=maze_layout, num_spawns=num_spawns, num_objects=num_objects
        )
        self.entity_layer = self.maze.entity_layer
        self.rows = len(self.entity_layer)
        self.cols = len(self.entity_layer[0])

        # 定义动作空间：4个离散动作（上、下、左、右）
        self.action_space = spaces.Discrete(4)

        # 定义观测空间：
        if self.pos_aware:
            # [current_row, current_col, target_row, target_col, up_cell, down_cell, left_cell, right_cell]
            self.observation_space = spaces.MultiDiscrete(
                [
                    self.rows,
                    self.cols,
                    self.rows,
                    self.cols,
                    *([OBS_DICT.__len__()] * 4),
                ]
            )
        else:
            # [up_cell, down_cell, left_cell, right_cell]
            self.observation_space = spaces.MultiDiscrete([*([OBS_DICT.__len__()] * 4)])

        # 环境状态
        self.current_pos = None
        self.target_pos = None
        self.steps = 0
        # 用于渲染的窗口
        self.window = None
        self.clock = None

    def _get_observation(self):
        """
        获取观测信息

        Returns:
            np.ndarray: 观测向量 [current_row, current_col, target_row, target_col,
                                 up_cell, down_cell, left_cell, right_cell]
        """
        row, col = self.current_pos
        target_row, target_col = self.target_pos

        # 获取相邻格子
        neighbors = {
            "up": self.entity_layer[row - 1][col] if row > 0 else "*",
            "down": self.entity_layer[row + 1][col] if row < self.rows - 1 else "*",
            "left": self.entity_layer[row][col - 1] if col > 0 else "*",
            "right": self.entity_layer[row][col + 1] if col < self.cols - 1 else "*",
        }

        if self.pos_aware:
            obs = np.array(
                [
                    row,
                    col,
                    target_row,
                    target_col,
                    encode_cell(neighbors["up"]),
                    encode_cell(neighbors["down"]),
                    encode_cell(neighbors["left"]),
                    encode_cell(neighbors["right"]),
                ],
                dtype=np.int32,
            )
        else:
            obs = np.array(
                [
                    encode_cell(neighbors["up"]),
                    encode_cell(neighbors["down"]),
                    encode_cell(neighbors["left"]),
                    encode_cell(neighbors["right"]),
                ],
                dtype=np.int32,
            )

        return obs

    def _get_info(self):
        """获取额外信息"""
        return {
            "steps": self.steps,
            "distance": abs(self.current_pos[0] - self.target_pos[0])
            + abs(self.current_pos[1] - self.target_pos[1]),
        }

    def reset(self, seed=None, options=None):
        """
        重置环境

        Args:
            seed: 随机种子
            options: 额外选项

        Returns:
            observation: 初始观测
            info: 额外信息
        """
        super().reset(seed=seed)

        # 重新生成迷宫（随机放置起点和目标）
        self.maze.regenerate()
        self.entity_layer = self.maze.entity_layer

        # 查找起点和目标位置
        spawn_positions, object_positions = find_positions(
            self.entity_layer, self.rows, self.cols
        )

        # 设置初始位置
        if spawn_positions:
            self.current_pos = spawn_positions[0]
        else:
            self.current_pos = (1, 1)

        # 设置目标位置
        if object_positions:
            self.target_pos = object_positions[0]
        else:
            self.target_pos = (self.rows - 2, self.cols - 2)

        # 重置步数
        self.steps = 0

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        执行动作

        Args:
            action: 动作(0=上, 1=下, 2=左, 3=右）

        Returns:
            observation: 新的观测
            reward: 奖励
                - 每步小负奖励 -0.1
                - 碰壁给负奖励 -1.0
                - 到达目标给正奖励 10.0
            terminated: 是否终止（到达目标）
            truncated: 是否截断（超过最大步数）, False, 由包装器处理
            info: 额外信息
        """
        self.steps += 1

        # 获取当前位置
        row, col = self.current_pos

        # 计算新位置
        new_row, new_col = row, col

        if action == Direction.UP.value:
            new_row = row - 1
        elif action == Direction.DOWN.value:
            new_row = row + 1
        elif action == Direction.LEFT.value:
            new_col = col - 1
        elif action == Direction.RIGHT.value:
            new_col = col + 1

        # 检查新位置是否有效
        if (
            0 <= new_row < self.rows
            and 0 <= new_col < self.cols
            and self.entity_layer[new_row][new_col] != "*"
        ):
            # 有效移动，更新位置
            self.current_pos = (new_row, new_col)
            reward = -0.1  # 每步小负奖励，鼓励快速完成任务
        else:
            # 无效移动，大负奖励
            reward = -1.0

        # 检查是否到达目标
        terminated = False
        if self.current_pos == self.target_pos:
            reward = 10.0  # 到达目标给正奖励
            terminated = True

        # 获取观测和信息
        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        """渲染环境"""
        if self.render_mode == "rgb_array":
            return render_frame(
                self.entity_layer,
                self.current_pos,
                self.render_mode,
                self.window,
                self.clock,
                self.metadata["render_fps"],
            )

    def _render_frame(self):
        """渲染一帧"""
        self.window, self.clock = render_frame(
            self.entity_layer,
            self.current_pos,
            self.render_mode,
            self.window,
            self.clock,
            self.metadata["render_fps"],
        )

    def close(self):
        """关闭环境"""
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


# 注册环境
from gymnasium.envs.registration import register

register(
    id="MazeNavigation-v0",
    entry_point="maze_env:MazeNavigationEnv",
    max_episode_steps=1000,
    kwargs={
        "pos_aware": True,
    },
)

register(
    id="MazeNavigation-v1",
    entry_point="maze_env:MazeNavigationEnv",
    max_episode_steps=1000,
    kwargs={
        "pos_aware": False,
    },
)


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    # 创建 imgs 文件夹
    os.makedirs("./imgs", exist_ok=True)

    # 创建环境
    env = gym.make("MazeNavigation-v0", render_mode="rgb_array")
    env_unwrapped = env.unwrapped

    print("=" * 60)
    print("环境分析")
    print("=" * 60)

    # 检查观测空间
    print("\n1. 观测空间分析:")
    print(f"   观测空间类型: {type(env.observation_space)}")
    print(f"   观测空间形状: {env.observation_space.shape}")
    print(f"   观测空间维度: {env.observation_space.nvec}")
    # 检查动作空间
    print("\n2. 动作空间分析:")
    print(f"   动作空间类型: {type(env.action_space)}")
    print(f"   动作空间大小: {env.action_space.n}")
    print(f"   动作定义: {[f'{d.name}:{d.value}' for d in Direction]}")

    # 测试多次 reset，验证起点和目标位置是否变化
    print("\n3. 验证每次 reset 都改变迷宫初始位置和目标位置:")
    positions_history = []
    for i in range(5):
        obs, info = env.reset()
        positions_history.append(
            {
                "episode": i + 1,
                "current_pos": env_unwrapped.current_pos,
                "target_pos": env_unwrapped.target_pos,
                "distance": info["distance"],
            }
        )
        print(
            f"   Episode {i + 1}: 起点={env_unwrapped.current_pos}, 目标={env_unwrapped.target_pos}, 距离={info['distance']}"
        )

    # 检查位置是否变化
    current_positions = [p["current_pos"] for p in positions_history]
    target_positions = [p["target_pos"] for p in positions_history]
    unique_current = len(set(current_positions))
    unique_target = len(set(target_positions))

    print(f"\n   起点位置变化: {unique_current} 个不同位置")
    print(f"   目标位置变化: {unique_target} 个不同位置")

    if unique_current > 1:
        print("   ✓ 起点位置在每次 reset 时发生变化")
    else:
        print("   ✗ 起点位置在每次 reset 时没有变化")

    if unique_target > 1:
        print("   ✓ 目标位置在每次 reset 时发生变化")
    else:
        print("   ✗ 目标位置在每次 reset 时没有变化")

    # 渲染并保存迷宫可视化
    print("\n4. 迷宫渲染可视化保存到 ./imgs 文件夹:")

    # 保存 5 个不同的迷宫布局
    for i in range(5):
        obs, info = env.reset()
        img = env.render()

        # 保存图像
        img_path = f"./imgs/maze_layout_{i + 1}.png"
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(
            f"Maze Layout {i + 1}\nStart: {env_unwrapped.current_pos}, Target: {env_unwrapped.target_pos}"
        )
        plt.axis("off")
        plt.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   保存: {img_path}")

    # 测试动作执行
    print("\n5. 测试动作执行:")
    obs, info = env.reset()
    print(f"   初始观测: {obs}")
    print(f"   初始位置: {env_unwrapped.current_pos}")
    print(f"   目标位置: {env_unwrapped.target_pos}")

    # 执行几个动作
    actions = [0, 1, 2, 3]
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    for action, action_name in zip(actions, action_names):
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"   执行动作 {action_name}({action}): 新位置={env_unwrapped.current_pos}, 奖励={reward:.1f}, 终止={terminated}, 截断={truncated}"
        )

    # 保存最终状态
    img = env.render()
    img_path = "./imgs/maze_final_state.png"
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(
        f"Final State\nPosition: {env_unwrapped.current_pos}, Target: {env_unwrapped.target_pos}"
    )
    plt.axis("off")
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   保存最终状态: {img_path}")

    print("\n" + "=" * 60)
    print("环境分析完成！")
    print("=" * 60)

    # 关闭环境
    env.close()
