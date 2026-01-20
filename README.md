# PMN：基于本体感觉的迷宫导航（Gymnasium 环境 + SB3 训练脚本）

**Languages / 语言**: 中文（本页） | [English](README_EN.md)

一个轻量的迷宫导航强化学习项目：

- 自定义 Gymnasium 环境 `MazeNavigationEnv`（基于 `labmaze` 生成随机起点/目标）
- 支持两种观测设置：带位置信息（`MazeNavigation-v0`）/ 不带位置信息（`MazeNavigation-v1`）
- 使用 Stable-Baselines3 / sb3-contrib 训练 `PPO` 与 `RecurrentPPO(MlpLstmPolicy)`
- 提供 pygame 可视化脚本（随机策略 / 模型策略 / 手动 WASD 控制）

> 备注：本仓库当前以脚本形式组织，适合直接跑实验并把 `logs/` 输出作为训练记录。

---

## 功能特性

- **环境**：离散动作（上/下/左/右），迷宫墙体/通路/起点/目标
- **奖励设计**：
  - 每步：`-0.1`
  - 碰壁：`-1.0`
  - 到达目标：`+10.0`（episode 终止）
- **环境注册**：
  - `MazeNavigation-v0`：`pos_aware=True`（观测包含当前位置与目标位置）
  - `MazeNavigation-v1`：`pos_aware=False`（仅局部邻域本体感觉）
- **训练**：向量化并行环境（默认 `n_envs=256`）+ EvalCallback
- **日志**：每次训练输出到 `./logs/YYYYMMDD_HHMMSS/`（csv/json/tensorboard 等）

---

## 目录结构

- `maze_env.py`：Gymnasium 环境实现与环境注册（`MazeNavigation-v0/v1`）
- `train.py`：训练入口（PPO / RecurrentPPO）
- `visualize_maze.py`：pygame 可视化（random/model/manual）
- `test_gym_env.py`：环境基本功能自检脚本
- `utils.py`：迷宫布局/编码/渲染工具
- `build_maze_example.py`：`labmaze` 迷宫构建示例
- `logs/`：训练输出（默认已在 `.gitignore` 中忽略）
- `imgs/`：环境渲染图片输出目录

---

## 环境依赖

Python 依赖见 `requirements.txt`：

```bash
pip install -r requirements.txt
```

可视化/人类渲染需要额外安装：

```bash
pip install pygame
```

> 说明：训练脚本通常不需要 `pygame`；只有在 `render_mode="human"` 或运行可视化脚本时才需要。

---

## 快速开始

### 1) 安装

```bash
pip install -r requirements.txt
```

### 2) 训练（PPO / RecurrentPPO）

`train.py` 会自动创建 `./logs/<timestamp>/`，并将模型保存为 `<model_name>.zip`。

- 训练 `RecurrentPPO`：

```bash
python train.py --env_id MazeNavigation-v1 --algorithm RecurrentPPO
```

- 训练 `PPO`（无记忆）：

```bash
python train.py --env_id MazeNavigation-v1 --algorithm PPO
```

常用可选参数（来自 `train.py --help`）：

- `--n_envs`：并行环境数（默认 256）
- `--n_eval_envs`：评估并行环境数（默认 64）
- `--total_timesteps`：总训练步数（默认 1,000,000）
- `--use_cpu`：强制使用 CPU
- `--log_dir`：日志根目录（默认 `./logs`）
- `--model_name`：最终保存的模型文件名（默认 `maze_model`）

### 3) TensorBoard 查看训练曲线

训练日志会写入同一目录的 tensorboard events：

```bash
tensorboard --logdir ./logs
```

---

## 可视化与交互

### 手动控制（WASD/方向键）

```bash
python visualize_maze.py --mode manual --render_fps 4
```

按键：

- `W/A/S/D` 或方向键：移动
- `R`：重置
- `ESC/Q`：退出

### 随机策略可视化

```bash
python visualize_maze.py --mode random --render_fps 30 --max_episodes 10 --max_steps 1000
```

### 加载训练好的 SB3 模型

```bash
python visualize_maze.py --mode model --model ./logs/<timestamp>/maze_model.zip --model_cls PPO
```

如果你训练的是带记忆的模型：

```bash
python visualize_maze.py --mode model --model ./logs/<timestamp>/maze_model.zip --model_cls RecurrentPPO
```

`visualize_maze.py` 关键参数：

- `--mode`：`random | model | manual`
- `--model`：模型路径（仅 `model` 模式）
- `--model_cls`：`PPO | RecurrentPPO`（仅 `model` 模式）
- `--pos_aware`：使用位置感知环境（会选用 `MazeNavigation-v0`）
- `--render_fps`：渲染帧率

---

## 环境自检

运行基础测试：

```bash
python test_gym_env.py
```
