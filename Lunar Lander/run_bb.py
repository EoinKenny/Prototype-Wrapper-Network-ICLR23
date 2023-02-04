from model import ActorCritic
import torch
import gym
from PIL import Image
import numpy as np   


n_episodes = 150
name='LunarLander_TWO.pth'
env = gym.make('LunarLander-v2')
policy = ActorCritic()
policy.load_state_dict(torch.load('./preTrained/{}'.format(name)))

all_rewards = list()

for i_episode in range(1, n_episodes+1):
    state = env.reset()
    running_reward = 0
    for t in range(10000):
        action, latent_x = policy(state)
        state, reward, done, _ = env.step(action)
        running_reward += reward   
        if done:
            break
    all_rewards.append(running_reward)
env.close()
            
data_rewards = np.array(all_rewards)

print("===== Data Reward:")
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt( len(data_rewards) )  )

