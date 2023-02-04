"""
Use final model to get the labelled data going forwards
"""

import gym
from tqdm import tqdm
from time import sleep
from model import DQN_Agent
import numpy as np      
import pdb


# Custom for each domain == Load in environment and model
env = gym.make("CartPole-v1").unwrapped
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
exp_replay_size = 256
agent = DQN_Agent(seed=1423, layer_sizes=[input_dim, 64, output_dim], lr=1e-3, sync_freq=5,
		  exp_replay_size=exp_replay_size)
agent.load_pretrained_model("weights/cartpole-dqn.pth")


X_train = list()
a_train = list()
q_train = list()
s_train = list()

# obs_train = np.load('data/obs_train.npy')

# reward_arr = []
# for i in tqdm(range(len(obs_train))):

#         A, latent_x, q_values = agent.get_action(obs_train[i], env.action_space.n, epsilon=0)
#         q_train.append(q_values.cpu().detach().tolist())
#         X_train.append(latent_x.cpu().detach().tolist())
#         a_train.append(A.cpu().detach().item())
#         # s_train.append(pix.tolist())



X_train = list()
a_train = list()
q_train = list()
obs_train = list()
all_pixels = list()

for i in tqdm(range(1000)):

	ep_obs = list()
	ep_a   = list()
	ep_x   = list()
	ep_q   = list()

	obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0

	while not done:

		ep_len += 1

		ep_obs.append(obs.tolist())

		A, x, q = agent.get_action(obs, env.action_space.n, 0)

		ep_a.append(A.item())
		ep_q.append(q.tolist())
		ep_x.append(x.tolist())

		obs_next, _, done, _ = env.step(A.item())
		obs = obs_next

		# env.render()

		if ep_len == 200:
			break


	obs_train.append(ep_obs)
	X_train.append(ep_x)
	a_train.append(ep_a)
	q_train.append(ep_q)



obs_train = np.array(obs_train)
np.save('data/obs_train.npy', obs_train)
print("Num instances produced:", len(obs_train))


X_train = np.array(X_train)
a_train = np.array(a_train)
q_train = np.array(q_train)

np.save('data/X_train.npy', X_train)
np.save('data/a_train.npy', a_train)
np.save('data/q_train.npy', q_train)




