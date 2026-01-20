# PMN: Proprioception-Based Maze Navigation (Gymnasium Env + SB3 Training Scripts)

A lightweight reinforcement learning project for maze navigation:

- Custom Gymnasium environment `MazeNavigationEnv` (random start/goal generated via `labmaze`)
- Two observation settings: position-aware (`MazeNavigation-v0`) / position-unaware (`MazeNavigation-v1`)
- Train with Stable-Baselines3 / sb3-contrib using `PPO` and `RecurrentPPO(MlpLstmPolicy)`
- Pygame visualization (random policy / model policy / manual WASD control)

> Note: This repo is currently organized as scripts, suitable for running experiments directly and using `logs/` as training records.

---

## Features

- **Environment**: discrete actions (up/down/left/right), maze walls/paths/start/goal
- **Reward shaping**:
  - Per step: `-0.1`
  - Hit wall: `-1.0`
  - Reach goal: `+10.0` (terminates episode)
- **Env registration**:
  - `MazeNavigation-v0`: `pos_aware=True` (observation includes current position and goal position)
  - `MazeNavigation-v1`: `pos_aware=False` (local proprioception only)
- **Training**: vectorized parallel envs (default `n_envs=256`) + EvalCallback
- **Logging**: each run writes to `./logs/YYYYMMDD_HHMMSS/` (csv/json/tensorboard, etc.)

---

## Project Structure

- `maze_env.py`: Gymnasium environment implementation and env registration (`MazeNavigation-v0/v1`)
- `train.py`: training entry (PPO / RecurrentPPO)
- `visualize_maze.py`: pygame visualization (random/model/manual)
- `test_gym_env.py`: basic environment sanity checks
- `utils.py`: maze layout/encoding/rendering utilities
- `build_maze_example.py`: `labmaze` construction example
- `logs/`: training outputs (ignored by default in `.gitignore`)
- `imgs/`: rendered images output directory

---

## Dependencies

Python dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Train (PPO / RecurrentPPO)

`train.py` automatically creates `./logs/<timestamp>/` and saves the model as `<model_name>.zip`.

- Train `RecurrentPPO`:

```bash
python train.py --env_id MazeNavigation-v1 --algorithm RecurrentPPO
```

- Train `PPO` (no memory):

```bash
python train.py --env_id MazeNavigation-v1 --algorithm PPO
```

Common optional arguments (from `train.py --help`):

- `--n_envs`: number of parallel envs (default 256)
- `--n_eval_envs`: number of parallel eval envs (default 64)
- `--total_timesteps`: total timesteps (default 1,000,000)
- `--use_cpu`: force CPU
- `--log_dir`: root log directory (default `./logs`)
- `--model_name`: output model name (default `maze_model`)

### 3) View Curves with TensorBoard

TensorBoard events are written in the same log directory:

```bash
tensorboard --logdir ./logs
```

---

## Visualization & Interaction

### Manual Control (WASD / Arrow Keys)

```bash
python visualize_maze.py --mode manual --render_fps 4
```

Keys:

- `W/A/S/D` or arrow keys: move
- `R`: reset
- `ESC/Q`: quit

### Random Policy Visualization

```bash
python visualize_maze.py --mode random --render_fps 30 --max_episodes 10 --max_steps 1000
```

### Load a Trained SB3 Model

```bash
python visualize_maze.py --mode model --model ./logs/<timestamp>/maze_model.zip --model_cls PPO
```

If you trained a recurrent model:

```bash
python visualize_maze.py --mode model --model ./logs/<timestamp>/maze_model.zip --model_cls RecurrentPPO
```

Key args in `visualize_maze.py`:

- `--mode`: `random | model | manual`
- `--model`: model path (model mode only)
- `--model_cls`: `PPO | RecurrentPPO` (model mode only)
- `--pos_aware`: use the position-aware env (selects `MazeNavigation-v0`)
- `--render_fps`: render FPS

---

## Environment Self-Test

Run basic tests:

```bash
python test_gym_env.py
```
