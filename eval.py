from stable_baselines3 import PPO
from make_env import MazeEnv
import time
import os

env = MazeEnv()
model = PPO.load("models/maze_cnn_final", env=env, device="cuda")

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)
    obs, reward, done, _ = env.step(action)
    os.system('clear')
    env.render()
    time.sleep(0.1)

successes = 0
total_steps = 0
for _ in range(100):
    obs = env.reset()
    done = False
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, done, _ = env.step(action)
        steps += 1
    total_steps += steps
    if reward == 1.0:
        successes += 1

print(f"Solved {successes}/100 episodes")
print(f"Average steps per episode: {total_steps / 100:.1f}")
