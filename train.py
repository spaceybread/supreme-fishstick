from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from make_env import MazeEnv
from MazeCNN import MazeCNN

env = MazeEnv()
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    ent_coef=0.3,
    learning_rate=0.0003,
    device="cuda",
    policy_kwargs=dict(features_extractor_class=MazeCNN, features_extractor_kwargs=dict(features_dim=128))
)
checkpoint = CheckpointCallback(save_freq=50_000, save_path="./models/", name_prefix="maze_cnn")
model.learn(total_timesteps=3_000_000, callback=checkpoint)
model.save("models/maze_cnn_final")
