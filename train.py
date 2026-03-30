from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from make_env import MazeEnv

env = MazeEnv()
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    ent_coef=0.3,
    learning_rate=0.0003,
    device="cuda"
)
checkpoint = CheckpointCallback(save_freq=50_000, save_path="./models/", name_prefix="maze_cnn")
model.learn(total_timesteps=3_000_000, callback=checkpoint)
model.save("models/maze_cnn_final")
