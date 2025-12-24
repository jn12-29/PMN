import argparse
import logging
import os
from datetime import datetime
import gymnasium as gym
import maze_env
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
import torch.nn as nn


def setup_logging(log_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, timestamp)
    os.makedirs(log_path, exist_ok=True)

    log_file = os.path.join(log_path, "info.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__), log_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO/RecurrentPPO agent on MazeNavigation environment"
    )

    parser.add_argument(
        "--env_id", type=str, default="MazeNavigation-v0", help="Environment ID"
    )
    parser.add_argument(
        "--n_envs", type=int, default=256, help="Number of parallel environments"
    )
    parser.add_argument(
        "--n_eval_envs",
        type=int,
        default=64,
        help="Number of parallel evaluation environments",
    )
    parser.add_argument(
        "--n_timesteps_per_iteration",
        type=int,
        default=12800,
        help="Time steps per iteration",
    )
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--use_cpu", action="store_true", default=False, help="Use CPU for training"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="RecurrentPPO",
        choices=["PPO", "RecurrentPPO"],
        help="Algorithm to use",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="Directory to save logs"
    )
    parser.add_argument(
        "--model_name", type=str, default="maze_model", help="Model name for saving"
    )
    parser.add_argument(
        "--tensorboard_log", type=str, default=None, help="Tensorboard log directory"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger, log_path = setup_logging(args.log_dir)

    logger.info("=" * 50)
    logger.info("Starting training with the following configuration:")
    logger.info(f"Environment: {args.env_id}")
    logger.info(f"Number of environments: {args.n_envs}")
    logger.info(f"Time steps per iteration: {args.n_timesteps_per_iteration}")
    logger.info(f"Number of epochs: {args.n_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Total timesteps: {args.total_timesteps}")
    logger.info(f"Use CPU: {args.use_cpu}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Log directory: {log_path}")
    logger.info("=" * 50)

    n_steps = args.n_timesteps_per_iteration // args.n_envs
    if n_steps <= 0:
        n_steps = 1
    device = "cpu" if args.use_cpu else "cuda"

    logger.info(
        f"Creating vectorized environment with {args.n_envs} parallel environments..."
    )
    env = make_vec_env(args.env_id, n_envs=args.n_envs)

    logger.info(f"Creating {args.algorithm} model...")
    if args.algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            verbose=1,
            device=device,
        )
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            n_steps=n_steps,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            verbose=1,
            device=device,
            policy_kwargs=dict(n_lstm_layers=2, activation_fn=nn.SiLU),
        )

    model.set_logger(
        configure(log_path, ["stdout", "log", "json", "csv", "tensorboard"])
    )

    logger.info(f"Model policy: {model.policy}")

    eval_callback = EvalCallback(
        make_vec_env(args.env_id, n_envs=args.n_eval_envs),
        best_model_save_path=log_path,
        log_path=log_path,
        eval_freq=100,
        deterministic=True,
        render=False,
    )

    logger.info(f"Starting training for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    model_path = os.path.join(log_path, f"{args.model_name}.zip")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    env.close()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
