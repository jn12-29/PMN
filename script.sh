CUDA_VISIBLE_DEVICES=0 python train.py --env_id MazeNavigation-v1 --algorithm RecurrentPPO

CUDA_VISIBLE_DEVICES=1 python train.py --env_id MazeNavigation-v1 --algorithm PPO
