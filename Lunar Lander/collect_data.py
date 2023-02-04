from model import ActorCritic
import torch
import gym
from PIL import Image
import numpy as np   

from collections import Counter


n_episodes = 30
name='LunarLander_TWO.pth'

X_train = list()
a_train = list()
obs_train = list()
q_train = list()

env = gym.make('LunarLander-v2')
policy = ActorCritic()
policy.load_state_dict(torch.load('./preTrained/{}'.format(name)))

render = False
save_gif = False
actions = list()

for i_episode in range(1, n_episodes+1):
    state = env.reset()
    running_reward = 0
    for t in range(10000):
        action, latent_x = policy(state)

        actions.append(action)

        state, reward, done, _ = env.step(action)
        running_reward += reward

        X_train.append(latent_x.detach().tolist())
        a_train.append(action)
        obs_train.append(state.tolist())

        if render:
             # env.render()
             if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))
                 
        if done:
            break

    print('Episode {}\tReward: {}'.format(i_episode, running_reward))
env.close()
            

X_train = np.array(X_train)
a_train = np.array(a_train)
obs_train = np.array(obs_train)

np.save('data/X_train.npy', X_train)
np.save('data/a_train.npy', a_train)
np.save('data/obs_train.npy', obs_train)

print("Num instances produced:", len(X_train))
print(Counter(actions))






