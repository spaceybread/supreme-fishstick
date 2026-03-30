from stable_baselines3 import PPO
from make_env import MazeEnv
import time
import os

env = MazeEnv()
model = PPO.load("models/maze_ppo_final", env=env)

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    os.system('clear')
    env.render()
    time.sleep(0.1)
