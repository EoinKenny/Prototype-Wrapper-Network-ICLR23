import toml
import numpy as np
import torch
import pickle

from argparse import ArgumentParser
from os.path import join
from games.carracing import RacingNet, CarRacing
from ppo import PPO
from torch.distributions import Beta
from tqdm import tqdm


CONFIG_FILE = "config.toml"
device = 'cpu'
NUM_EPISODES = 30
 

def load_config():
	with open(CONFIG_FILE, "r") as f:
		config = toml.load(f)
	return config


cfg = load_config()
env = CarRacing(frame_skip=0, frame_stack=4)
net = RacingNet(env.observation_space.shape, env.action_space.shape)
ppo = PPO(
	env,
	net,
	lr=cfg["lr"],
	gamma=cfg["gamma"],
	batch_size=cfg["batch_size"],
	gae_lambda=cfg["gae_lambda"],
	clip=cfg["clip"],
	value_coef=cfg["value_coef"],
	entropy_coef=cfg["entropy_coef"],
	epochs_per_step=cfg["epochs_per_step"],
	num_steps=cfg["num_steps"],
	horizon=cfg["horizon"],
	save_dir=cfg["save_dir"],
	save_interval=cfg["save_interval"],
)

ppo.load("weights/agent_weights.pt")

states, real_actions, rewards, X_train = [], [], [], []
self_state = ppo._to_tensor(env.reset())
reward_arr = list()

for ep in tqdm(range(NUM_EPISODES)):
	next_state = ppo.env.reset()
	rew = 0
	done = False

	while not done:

		# Run one step of the environment based on the current policy
		value, alpha, beta, x = ppo.net(self_state)
		value, alpha, beta = value.squeeze(0), alpha.squeeze(0), beta.squeeze(0)
		policy = Beta(alpha, beta)

		# Choose how to get actions (sample or take mean)
		input_action = policy.mean.detach()
		# input_action = policy.sample()

		next_state, reward, done, info, real_action = ppo.env.step(input_action.cpu().numpy())
		next_state = ppo._to_tensor(next_state)

		self_state = next_state
		rew += reward

	reward_arr.append(rew)


data_rewards = np.array(reward_arr)

print("===== Data Reward:")
print("Rewards:", data_rewards)
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt(NUM_EPISODES)  )


