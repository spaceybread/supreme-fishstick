from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from make_env import MazeEnv
from MazeCNN import MazeCNN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

def make_env():
    return MazeEnv()

if __name__ == '__main__':
    num_envs = 16 
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    env = VecMonitor(env)

    model = PPO(
        "CnnPolicy",
        env,
        n_steps=2048, 
        batch_size=1024,
        verbose=1,
        n_epochs=5,    
        ent_coef=0.1,
        learning_rate=0.0003,
        device="cuda",
        policy_kwargs=dict(
            features_extractor_class=MazeCNN, 
            features_extractor_kwargs=dict(features_dim=128)
        )
    )

    checkpoint = CheckpointCallback(
        save_freq=200_000 // num_envs,
        save_path="./models/", 
        name_prefix="maze_cnn"
    )
    
    model.learn(total_timesteps=10_000_000, callback=checkpoint)
    model.save("models/maze_cnn_final")