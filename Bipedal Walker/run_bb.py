import gym
import numpy as np

from TD3 import TD3
from PIL import Image


env_name = "BipedalWalker-v3"
random_seed = 0
n_episodes = 30
lr = 0.002
max_timesteps = 2000
filename = "TD3_{}_{}".format(env_name, random_seed)
filename += '_solved'
directory = "./preTrained/{}".format(env_name)
env = gym.make(env_name, hardcore=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
policy = TD3(lr, state_dim, action_dim, max_action)
policy.load_actor(directory, filename)
NUM_ITERATIONS = 150

total_reward = list()
for ep in range(NUM_ITERATIONS):
    ep_reward = 0
    state = env.reset()

    for t in range(max_timesteps):
        A, x = policy.select_action(state)
        state, reward, done, _ = env.step(A)
        ep_reward += reward
        if done:
            break
    print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
    total_reward.append(ep_reward)
    ep_reward = 0
env.close()        
               
data_rewards = np.array(total_reward)

print("===== Data Reward:")
print("Rewards:", data_rewards)
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt(NUM_ITERATIONS)  )







