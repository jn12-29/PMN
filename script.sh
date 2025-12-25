CUDA_VISIBLE_DEVICES=0 python train.py --env_id MazeNavigation-v1 --algorithm RecurrentPPO

CUDA_VISIBLE_DEVICES=0 python train.py --env_id MazeNavigation-v0 --algorithm RecurrentPPO

CUDA_VISIBLE_DEVICES=0 python train.py --env_id MazeNavigation-v1 --algorithm PPO

CUDA_VISIBLE_DEVICES=0 python train.py --env_id MazeNavigation-v0 --algorithm PPO

python visualize_maze.py --mode manual --render_fps 4

python visualize_maze.py --mode random --render_fps 200



# v0 ppo
python visualize_maze.py --mode model --model ./logs/20251225_002654/maze_model.zip --pos_aware --model_cls PPO --render_fps 4

# v1 ppo
python visualize_maze.py --mode model --model ./logs/20251225_003120/maze_model.zip --model_cls PPO --render_fps 4

# v0 recurrentppo
python visualize_maze.py --mode model --model ./logs/20251224_170916/maze_model.zip --model_cls RecurrentPPO

# v1 recurrentppo
python visualize_maze.py --mode model --model ./logs/20251224_170916/maze_model.zip --model_cls RecurrentPPO
