import pandas as pd
import os, time
from discrete_env import DiscretePortfolioEnv
from stable_baselines3 import DQN

returns_df = pd.read_csv('returns_dataset.csv', parse_dates=['Date'])
env = DiscretePortfolioEnv(returns_df, episode_length=500)

start_time_id = int(time.time()) # seconds since January 1, 1970, 00:00:00 (UTC)
models_dir = f'models/DQN-{start_time_id}'
log_dir = 'logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Train model - save and log along the way
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10_000 # number of timesteps between saves
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f'DQN-{start_time_id}')
    model.save(f'{models_dir}/{TIMESTEPS*i}')