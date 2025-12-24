#!/usr/bin/env python3
"""
测试 MazeNavigationEnv 环境的基本功能
"""

import gymnasium as gym
import numpy as np
from maze_env import MazeNavigationEnv


def test_basic_functionality():
    """测试环境的基本功能"""
    print("=== 测试环境基本功能 ===\n")

    # 创建环境
    env = MazeNavigationEnv(render_mode=None)

    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")
    print()

    # 测试 reset
    print("=== 测试 reset ===")
    obs, info = env.reset(seed=42)
    print(f"初始观测: {obs}")
    print(f"初始信息: {info}")
    print(f"当前位置: {env.current_pos}")
    print(f"目标位置: {env.target_pos}")
    print()

    # 测试 step
    print("=== 测试 step ===")
    action = 0  # 向上
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"动作: 向上")
    print(f"新观测: {obs}")
    print(f"奖励: {reward}")
    print(f"终止: {terminated}")
    print(f"截断: {truncated}")
    print(f"信息: {info}")
    print(f"新位置: {env.current_pos}")
    print()

    # 测试多个动作
    print("=== 测试多个动作 ===")
    actions = [1, 2, 3, 1, 1]  # 下、左、右、下、下
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        action_names = ["上", "下", "左", "右"]
        print(
            f"步骤 {i+1}: 动作={action_names[action]}, 位置={env.current_pos}, 奖励={reward:.2f}"
        )

        if terminated:
            print(f"🎉 到达目标！")
            break
        if truncated:
            print(f"⚠️ 超过最大步数")
            break
    print()

    # 关闭环境
    env.close()
    print("=== 测试完成 ===")


def test_random_agent():
    """测试随机智能体"""
    print("\n=== 测试随机智能体 ===\n")

    env = MazeNavigationEnv(render_mode=None)

    num_episodes = 5

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0

        print(f"回合 {episode + 1}:")

        while True:
            action = env.action_space.sample()  # 随机动作
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

        print(
            f"  步数: {steps}, 总奖励: {episode_reward:.2f}, "
            f"状态: {'成功' if terminated else '失败'}"
        )

    env.close()
    print("\n=== 随机智能体测试完成 ===")


def test_observation_space():
    """测试观测空间的范围"""
    print("\n=== 测试观测空间 ===\n")

    env = MazeNavigationEnv(render_mode=None)

    # 收集多个观测样本
    observations = []
    for _ in range(100):
        obs, _ = env.reset()
        observations.append(obs)

        # 随机执行几步
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)

            if terminated or truncated:
                break

    observations = np.array(observations)

    print(f"观测形状: {observations.shape}")
    print()

    env.close()
    print("\n=== 观测空间测试完成 ===")


if __name__ == "__main__":
    test_basic_functionality()
    test_random_agent()
    test_observation_space()
    print("\n✅ 所有测试通过！")
