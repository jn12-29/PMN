#!/usr/bin/env python3
"""
迷宫可视化脚本
支持三种模式：
1. 随机策略
2. SB3模型策略
3. 手动控制（WASD）
"""

import argparse
import gymnasium as gym
import pygame
import numpy as np
from maze_env import MazeNavigationEnv, Direction


class MazeVisualizer:
    def __init__(self, mode="random", model_path=None, model_cls=None, pos_aware=True):
        """
        初始化可视化器

        Args:
            mode: 模式 ("random", "model", "manual")
            model_path: SB3模型路径（仅model模式需要）
            model_cls: SB3模型类名，如 "PPO", "DQN", "A2C", "SAC", "TD3"（仅model模式需要）
            pos_aware: 是否使用位置感知环境
        """
        self.mode = mode
        self.model_path = model_path
        self.model_cls = model_cls
        self.pos_aware = pos_aware

        env_id = "MazeNavigation-v0" if pos_aware else "MazeNavigation-v1"
        self.env = gym.make(env_id, render_mode="human")
        self.env_unwrapped = self.env.unwrapped

        self.model = None
        if mode == "model":
            self._load_model()

        self.running = True
        self.episode = 0

    def _load_model(self):
        """加载SB3模型"""
        try:
            from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
            from sb3_contrib import RecurrentPPO

            if self.model_path is None:
                raise ValueError("模型路径不能为空")

            if self.model_cls is None:
                raise ValueError("模型类不能为空，请使用 --model_cls 参数指定")

            model_classes = {
                "PPO": PPO,
                "DQN": DQN,
                "A2C": A2C,
                "SAC": SAC,
                "TD3": TD3,
                "RecurrentPPO": RecurrentPPO,
            }
            if self.model_cls not in model_classes:
                raise ValueError(
                    f"不支持的模型类: {self.model_cls}，支持的类: {list(model_classes.keys())}"
                )

            self.model = model_classes[self.model_cls].load(self.model_path)
            print(f"成功加载 {self.model_cls} 模型: {self.model_path}")
        except ImportError:
            raise ImportError("请安装 stable-baselines3: pip install stable-baselines3")

    def _get_random_action(self):
        """获取随机动作"""
        return self.env.action_space.sample()

    def _get_model_action(self, obs):
        """获取模型预测的动作"""
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def _get_manual_action(self):
        """获取手动输入的动作"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    return Direction.UP.value
                elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    return Direction.DOWN.value
                elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    return Direction.LEFT.value
                elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    return Direction.RIGHT.value
                elif event.key == pygame.K_r:
                    return "reset"
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    self.running = False
                    return None
        return None

    def run_random(self, max_episodes=10, max_steps=1000):
        """运行随机策略"""
        print("=" * 60)
        print("模式: 随机策略")
        print("=" * 60)
        print("按 ESC 或关闭窗口退出")
        print("=" * 60)

        for self.episode in range(1, max_episodes + 1):
            if not self.running:
                break

            obs, info = self.env.reset()
            print(f"\nEpisode {self.episode}")
            print(f"  起点: {self.env_unwrapped.current_pos}")
            print(f"  目标: {self.env_unwrapped.target_pos}")

            step = 0
            for step in range(1, max_steps + 1):
                if not self.running:
                    break

                action = self._get_random_action()
                obs, reward, terminated, truncated, info = self.env.step(action)

                if terminated or truncated:
                    print(
                        f"  步数: {step}, 奖励: {reward:.1f}, 状态: {'成功' if terminated else '超时'}"
                    )
                    break

            if not self.running:
                break

        self.env.close()

    def run_model(self, max_episodes=10, max_steps=1000):
        """运行SB3模型策略"""
        print("=" * 60)
        print("模式: SB3模型策略")
        print("=" * 60)
        print(f"模型: {self.model_path}")
        print("按 ESC 或关闭窗口退出")
        print("=" * 60)

        for self.episode in range(1, max_episodes + 1):
            if not self.running:
                break

            obs, info = self.env.reset()
            print(f"\nEpisode {self.episode}")
            print(f"  起点: {self.env_unwrapped.current_pos}")
            print(f"  目标: {self.env_unwrapped.target_pos}")

            step = 0
            total_reward = 0
            for step in range(1, max_steps + 1):
                if not self.running:
                    break

                action = self._get_model_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward

                if terminated or truncated:
                    print(
                        f"  步数: {step}, 总奖励: {total_reward:.1f}, 状态: {'成功' if terminated else '超时'}"
                    )
                    break

            if not self.running:
                break

        self.env.close()

    def run_manual(self, max_episodes=100):
        """运行手动控制模式"""
        print("=" * 60)
        print("模式: 手动控制")
        print("=" * 60)
        print("控制说明:")
        print("  W / ↑ : 向上移动")
        print("  S / ↓ : 向下移动")
        print("  A / ← : 向左移动")
        print("  D / → : 向右移动")
        print("  R     : 重置迷宫")
        print("  Q / ESC: 退出")
        print("=" * 60)

        obs, info = self.env.reset()
        print(f"\n起点: {self.env_unwrapped.current_pos}")
        print(f"目标: {self.env_unwrapped.target_pos}")

        while self.running:
            action = self._get_manual_action()

            if action is None:
                continue

            if action == "reset":
                obs, info = self.env.reset()
                self.episode += 1
                print(f"\nEpisode {self.episode}")
                print(f"  起点: {self.env_unwrapped.current_pos}")
                print(f"  目标: {self.env_unwrapped.target_pos}")
                continue

            obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated:
                print(f"  成功到达目标！")
                obs, info = self.env.reset()
                self.episode += 1
                print(f"\nEpisode {self.episode}")
                print(f"  起点: {self.env_unwrapped.current_pos}")
                print(f"  目标: {self.env_unwrapped.target_pos}")

        self.env.close()

    def run(self, max_episodes=10, max_steps=1000):
        """运行可视化"""
        if self.mode == "random":
            self.run_random(max_episodes, max_steps)
        elif self.mode == "model":
            self.run_model(max_episodes, max_steps)
        elif self.mode == "manual":
            self.run_manual(max_episodes)
        else:
            raise ValueError(f"未知模式: {self.mode}")


def main():
    parser = argparse.ArgumentParser(description="迷宫可视化工具")
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["random", "model", "manual"],
        help="运行模式: random(随机), model(SB3模型), manual(手动控制)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="SB3模型路径（仅model模式需要）",
    )
    parser.add_argument(
        "--model_cls",
        type=str,
        default=None,
        choices=["PPO", "DQN", "A2C", "SAC", "TD3"],
        help="SB3模型类名（仅model模式需要）: PPO, DQN, A2C, SAC, TD3",
    )
    parser.add_argument(
        "--pos_aware",
        action="store_true",
        default=False,
        help="使用位置感知环境（默认True）",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=10,
        help="最大回合数（仅random和model模式）",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="每回合最大步数（仅random和model模式）",
    )

    args = parser.parse_args()

    visualizer = MazeVisualizer(
        mode=args.mode,
        model_path=args.model,
        model_cls=args.model_cls,
        pos_aware=args.pos_aware,
    )

    visualizer.run(
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
