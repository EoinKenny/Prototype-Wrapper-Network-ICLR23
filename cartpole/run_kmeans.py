import gym
import torch 
import torch.nn as nn
import numpy as np  

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN, OPTICS

from random import sample
from tqdm import tqdm
from time import sleep
from model import DQN_Agent
from collections import Counter
from scipy.spatial.distance import euclidean
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans, DBSCAN, OPTICS


NUM_CLASSES = 2
NUM_PROTOTYPES = 2
LATENT_SIZE = 64
PROTOTYPE_SIZE = 64
BATCH_SIZE = 64
NUM_EPOCHS = 30  # training epoch number
DEVICE = 'cpu'
NUM_SIMULATIONS = 30  # how many simulations for each trained model
NUM_ITERATIONS = 5  # how many datapoints



# Environment and Agent
env = gym.make("CartPole-v1").unwrapped
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
exp_replay_size = 256
agent = DQN_Agent(seed=1423, layer_sizes=[input_dim, 64, output_dim], lr=1e-3,
				  sync_freq=5, exp_replay_size=exp_replay_size)
agent.load_pretrained_model("weights/cartpole-dqn.pth")

X_train = np.load('data/X_train.npy', ).reshape(-1, LATENT_SIZE)
a_train = np.load('data/a_train.npy', ).flatten()
tensor_x = torch.Tensor(X_train)
tensor_y = torch.tensor(a_train)
train_dataset = TensorDataset(tensor_x, tensor_y)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)


def evaluate_loader(model, loader, cce_loss, intuition_labels=False):
	model.eval()
	total_correct = 0
	total_loss = 0
	total = 0
	
	with torch.no_grad():
		for i, data in enumerate(loader):
			imgs, labels = data
			
			if intuition_labels:
				labels = intuition_loss(observations)

			imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)            
			logits = model(imgs)
			loss = cce_loss(logits, labels)
			preds = torch.argmax(logits, dim=1)
			total_correct += sum(preds == labels).item()
			total += len(preds)
			total_loss += loss.item()
				
	return (total_correct / total) * 100, (total_loss / len(loader))


class PW_Net(nn.Module):

	def __init__(self, prototypes):
		super(PW_Net, self).__init__()
		self.prototypes = nn.Parameter(torch.tensor(prototypes, dtype=torch.float32), requires_grad=False)
		self.epsilon = 1e-5
		self.linear = nn.Linear(NUM_PROTOTYPES, NUM_CLASSES, bias=False) 
		self.softmax = nn.Softmax(dim=1)
		
	def __proto_layer_l2(self, x, p):
		output = list()
		b_size = x.shape[0]
		p = p.view(1, PROTOTYPE_SIZE).tile(b_size, 1).to(DEVICE) 
		c = x.view(b_size, PROTOTYPE_SIZE).to(DEVICE)      
		l2s = ( (c - p)**2 ).sum(axis=1).to(DEVICE) 
		act = torch.log( (l2s + 1. ) / (l2s + self.epsilon) ).to(DEVICE)  
		return act
	
	def __output_act_func(self, p_acts):        
		return self.softmax(p_acts)

	def __transforms(self, x):
		p_acts = list()
		for i in range(NUM_PROTOTYPES):
			action_prototype = self.prototypes[i]
			p_acts.append( self.__proto_layer_l2( x, action_prototype).view(-1, 1) )
		return torch.cat(p_acts, axis=1)
	
	def forward(self, x):
		p_acts = self.__transforms(x)    
		logits = self.linear(p_acts)                     
		final_outputs = self.__output_act_func(logits)
		return final_outputs


def trans_human_concepts(model, nn_human_x):
	model.eval()
	trans_nn_human_x = list()
	for i, t in enumerate(model.ts):
		trans_nn_human_x.append( t( torch.tensor(nn_human_x[i], dtype=torch.float32).view(1, -1)) )
	model.train()
	return torch.cat(trans_nn_human_x, dim=0)




final_accs = list()
final_rewards = list()

for _ in range(NUM_ITERATIONS):

	# Get prototypes with kmeans
	prototypes = list()
	for i in range(NUM_CLASSES):
		idxs = a_train == i
		temp_x = X_train[idxs]
		kmeans = KMeans(n_clusters=1, random_state=0).fit(temp_x)
		prototypes.append(kmeans.cluster_centers_.tolist()[0])
	prototypes = np.array(prototypes)

	#### Train Wrapper
	model = PW_Net(prototypes).eval()

	mse_loss = nn.MSELoss()
	cce_loss = nn.CrossEntropyLoss()




	#### Evaluate
	accuracy = list()
	reward_arr = []

	for i in tqdm(range(NUM_SIMULATIONS)):
		obs, done, rew = env.reset(), False, 0
		count = 0
		while not done and count < 200:
			AgentAction, latent_x, _ = agent.get_action(obs, env.action_space.n, epsilon=0)
			A = model( latent_x.view(1,-1) )
			A = torch.argmax(A).item()

			# print(env.step(A))

			obs, reward, done, info, _ = env.step(A)
			rew += reward
			count += 1
			accuracy.append( AgentAction.item() == A )
		reward_arr.append(count)

	print("average reward per episode :", sum(reward_arr) / len(reward_arr))
	sum(accuracy) / len(accuracy)

	final_accs.append( sum(accuracy) / len(accuracy) )
	final_rewards.append( sum(reward_arr) / len(reward_arr) )




data_errors = np.array(final_accs)
data_rewards = np.array(final_rewards)

print(" ")
print("===== Data Accuracy:")
print("Mean:", data_errors.mean())
print("Standard Error:", data_errors.std() / np.sqrt(NUM_ITERATIONS)  )
print(" ")
print("===== Data Reward:")
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt(NUM_ITERATIONS)  )



