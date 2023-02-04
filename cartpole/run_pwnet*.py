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

from sklearn.neighbors import KNeighborsRegressor



MODEL_DIR = 'weights/ppnet.pth'
NUM_CLASSES = 2
NUM_PROTOTYPES = 2
LATENT_SIZE = 64
PROTOTYPE_SIZE = 16
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
			logits, _ = model(imgs)
			loss = cce_loss(logits, labels)
			preds = torch.argmax(logits, dim=1)
			total_correct += sum(preds == labels).item()
			total += len(preds)
			total_loss += loss.item()
				
	return (total_correct / total) * 100


class PPNet(nn.Module):

	def __init__(self):
		super(PPNet, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(LATENT_SIZE, PROTOTYPE_SIZE),
			nn.BatchNorm1d(PROTOTYPE_SIZE),
			nn.ReLU(),
			nn.Linear(PROTOTYPE_SIZE, PROTOTYPE_SIZE),
		)
		prototypes = torch.randn( (NUM_PROTOTYPES, PROTOTYPE_SIZE), dtype=torch.float32 )
		self.prototypes = nn.Parameter(prototypes, requires_grad=True)
		self.epsilon = 1e-5
		self.linear = nn.Linear(NUM_PROTOTYPES, NUM_CLASSES, bias=False) 
		self.__make_linear_weights()
		self.softmax = nn.Softmax(dim=1)
		
	def __make_linear_weights(self):
		prototype_class_identity = torch.zeros(NUM_PROTOTYPES, NUM_CLASSES)
		num_prototypes_per_class = NUM_PROTOTYPES // NUM_CLASSES
		
		for j in range(NUM_PROTOTYPES):
			prototype_class_identity[j, j // num_prototypes_per_class] = 1
			
		positive_one_weights_locations = torch.t(prototype_class_identity)
		negative_one_weights_locations = 1 - positive_one_weights_locations

		incorrect_strength = .0
		correct_class_connection = 1
		incorrect_class_connection = incorrect_strength
		self.linear.weight.data.copy_(
			correct_class_connection * positive_one_weights_locations
			+ incorrect_class_connection * negative_one_weights_locations)
		
	def __proto_layer_l2(self, x):
		output = list()
		b_size = x.shape[0]
		p = self.prototypes.T.view(1, PROTOTYPE_SIZE, NUM_PROTOTYPES).tile(b_size, 1, 1).to(DEVICE) 
		c = x.view(b_size, PROTOTYPE_SIZE, 1).tile(1, 1, NUM_PROTOTYPES).to(DEVICE)            
		l2s = ( (c - p)**2 ).sum(axis=1).to(DEVICE) 
		act = torch.log( (l2s + 1. ) / (l2s + self.epsilon) ).to(DEVICE)   
		return act, l2s
	
	def __output_act_func(self, p_acts):        
		return self.softmax(p_acts)

	def forward(self, x): 
		
		# Transform
		x = self.main(x)
		
		# Prototype layer
		p_acts, l2s = self.__proto_layer_l2(x)
		
		# Linear Layer
		logits = self.linear(p_acts)
								
		# Activation Functions
		final_outputs = self.__output_act_func(logits)
		
		return final_outputs, x



def clust_loss(x, y, model, criterion):
	"""
	Forces each datapoint of a certain class to get closer to its prototype
	"""
	
	p = model.prototypes  # take prototypes in new feature space
	model = model.eval()
	x = model.main(x)  # transform into new feature space
	for idx, i in enumerate(Counter(y.cpu().numpy()).keys()):
		x_sub = x[y==i]
		target = p[i].repeat(len(x_sub), 1) 
		if idx == 0:
			loss = criterion(x_sub, target) 
		else:
			loss += criterion(x_sub, target)  
	model = model.train()
	return loss


def sep_loss(x, y, model, criterion):
	"""
	Take the distance of each training instance to each prototype NOT of its own class
	Sums them up and returns a negative distance to minimize
	"""
	
	p = model.prototypes  # take prototypes in new feature space
	model = model.eval()
	x = model.main(x)  # transform into new feature space
	loss = criterion(x, x)
	# Iterate each prototype
	for idx1, i in enumerate(Counter(y.cpu().numpy()).keys()):
		# select all training data aligned with that prototype
		x_sub = x[y==i]
		# Iterate all other prototypes
		for idx2, j in enumerate(Counter(y.cpu().numpy()).keys()):
			if i == j:
				continue
			# Select other prototype
			target = p[j].repeat(len(x_sub), 1) 
			# Take distance loss of training data to other prototypes
			loss += criterion(x_sub, target)
	model = model.train()
	return -loss / len(Counter(y.cpu().numpy()).keys())**2




final_accs = list()
final_rewards = list()

for _ in range(NUM_ITERATIONS):


	#### Train Wrapper
	model = PPNet().eval()
	mse_loss = nn.MSELoss()
	cce_loss = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-8)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
	best_acc = 0.
	model.train()

	# Freeze Linear Layer to make more interpretable
	model.linear.weight.requires_grad = False

	# Could tweak these, haven't tried
	lambda1 = 1.0
	lambda2 = 0.8
	lambda3 = 0.08

	for epoch in range(NUM_EPOCHS):

		model.eval()
		current_acc = evaluate_loader(model, train_loader, cce_loss)
		model.train()

		if current_acc > best_acc:
			torch.save(model.state_dict(), MODEL_DIR)
			best_acc = current_acc

		for instances, labels in train_loader:

			optimizer.zero_grad()

			instances, labels = instances.to(DEVICE), labels.to(DEVICE)
			logits, _ = model(instances)

			loss1 = cce_loss(logits, labels) * lambda1
			loss2 = clust_loss(instances, labels, model, mse_loss) * lambda2
			loss3 = sep_loss(instances, labels, model, mse_loss) * lambda3
			loss  = loss1 + loss2 + loss3

			loss.backward()
			optimizer.step()

		print("Acc:", current_acc, "       Epoch:", epoch)

		scheduler.step()


	#### Simulations
	model = PPNet().eval()
	model.load_state_dict(torch.load(MODEL_DIR))

	#### Project
	model = PPNet().eval()
	model.load_state_dict(torch.load(MODEL_DIR))
	trans_x = list()
	model.eval()
	with torch.no_grad():    
		for i in tqdm(range(len(X_train))):
			img = X_train[i]
			temp = model.main( torch.tensor(img.reshape(1, -1), dtype=torch.float32) )
			trans_x.append(temp[0].tolist())
	trans_x = np.array(trans_x)

	nn_xs = list()
	nn_as = list()
	nn_human_images = list()
	for i in range(NUM_PROTOTYPES):
		trained_prototype = model.prototypes.clone().detach()[i].view(1,-1)
		temp_x_train = trans_x
		knn = KNeighborsRegressor(algorithm='brute')
		knn.fit(temp_x_train, list(range(len(temp_x_train))))
		dist, nn_idx = knn.kneighbors(X=trained_prototype, n_neighbors=1, return_distance=True)
		nn_x = temp_x_train[nn_idx.item()]    
		nn_xs.append(nn_x.tolist())

	real_trans_x = nn_xs
	real_trans_x = torch.tensor( real_trans_x, dtype=torch.float32 )
	model.prototypes = torch.nn.Parameter(torch.tensor(real_trans_x, dtype=torch.float32))



	accuracy = list()
	reward_arr = []
	for i in tqdm(range(NUM_SIMULATIONS)):
		obs, done, rew = env.reset(), False, 0
		count = 0
		while not done and count < 200:
			AgentAction, latent_x, _ = agent.get_action(obs, env.action_space.n, epsilon=0)
			A, _ = model( latent_x.view(1,-1) )
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



