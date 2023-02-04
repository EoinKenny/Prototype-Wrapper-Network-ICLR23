import gym
from tqdm import tqdm
from time import sleep
from model import DQN_Agent
import numpy as np      
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.metrics import accuracy_score



NUM_ITERATIONS = 5
NUM_SIMULATIONS = 30


env = gym.make("CartPole-v1").unwrapped
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
exp_replay_size = 256
agent = DQN_Agent(seed=1423, layer_sizes=[input_dim, 64, output_dim], lr=1e-3, sync_freq=5,
                  exp_replay_size=exp_replay_size)
agent.load_pretrained_model("weights/cartpole-dqn.pth")


# load the model from disk
filename = 'weights/viper_dt.sav'
viper_dt = pickle.load(open(filename, 'rb'))



reward_all = list()
acc_all = list()

for _ in range(NUM_ITERATIONS):

    reward_arr = list()
    acc_arr = list()

    for i in tqdm(range(NUM_SIMULATIONS)):
        obs, done, rew = env.reset(), False, 0
        count = 0
        acc = 0

        while not done and count < 200:

            AgentAction, latent_x, _ = agent.get_action(obs, env.action_space.n, epsilon=0)
            A = viper_dt.predict( [obs] )

            obs, reward, done, info, _ = env.step(A.item())

            rew += reward
            # sleep(0.01)
            # env.render()
            count += 1
            acc += AgentAction.item() == A[0]



        reward_arr.append( count )
        acc_arr.append( acc / count )

    reward_all.append( sum(reward_arr) / len(reward_arr) )
        


data_errors = np.array(acc_arr)
data_rewards = np.array(reward_all)

print(" ")
print("===== Data Accuracy:")
print("Mean:", data_errors.mean())
print("Standard Error:", data_errors.std() / np.sqrt(NUM_ITERATIONS)  )
print(" ")
print("===== Data Reward:")
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt(NUM_ITERATIONS)  )




