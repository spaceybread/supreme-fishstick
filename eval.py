from stable_baselines3 import PPO
from make_env import MazeEnv
from tqdm import tqdm
import time
import os

env = MazeEnv()
model = PPO.load("models/maze_cnn_500000_steps.zip", env=env, device="cuda")

obs, info = env.reset()
done = False
truncated = False

while not (done or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(int(action)) 

    os.system('cls' if os.name == 'nt' else 'clear')
    env.render()
    time.sleep(0.1)

successes = 0
total_steps = 0
num_episodes = 100

for _ in tqdm(range(num_episodes)):
    obs, info = env.reset()
    done = False
    truncated = False
    steps = 0
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        steps += 1
        
        if done and reward == 1.0:
            successes += 1
            
    total_steps += steps

print(f"\nResults for maze_size {env.maze_size}:")
print(f"Solved {successes}/{num_episodes} episodes")
print(f"Average steps per episode: {total_steps / num_episodes:.1f}")