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
			logits = model(imgs)
			loss = cce_loss(logits, labels)
			preds = torch.argmax(logits, dim=1)
			total_correct += sum(preds == labels).item()
			total += len(preds)
			total_loss += loss.item()
				
	return (total_correct / total) * 100, (total_loss / len(loader))


class ListModule(object):
	#Should work with all kind of module
	def __init__(self, module, prefix, *args):
		self.module = module
		self.prefix = prefix
		self.num_module = 0
		for new_module in args:
			self.append(new_module)

	def append(self, new_module):
		if not isinstance(new_module, nn.Module):
			raise ValueError('Not a Module')
		else:
			self.module.add_module(self.prefix + str(self.num_module), new_module)
			self.num_module += 1

	def __len__(self):
		return self.num_module

	def __getitem__(self, i):
		if i < 0 or i >= self.num_module:
			raise IndexError('Out of bound')
		return getattr(self.module, self.prefix + str(i))


class PWNet(nn.Module):

	def __init__(self):
		super(PWNet, self).__init__()
		self.ts = ListModule(self, 'ts_')
		for i in range(NUM_PROTOTYPES):
			transformation = nn.Sequential(
				nn.Linear(LATENT_SIZE, PROTOTYPE_SIZE),
				nn.InstanceNorm1d(PROTOTYPE_SIZE),
				nn.ReLU(),
				nn.Linear(PROTOTYPE_SIZE, PROTOTYPE_SIZE),
			)
			self.ts.append(transformation)  
		self.prototypes = None
		self.epsilon = 1e-5
		self.linear = nn.Linear(NUM_PROTOTYPES, NUM_CLASSES, bias=False) 
		self.__make_linear_weights()
		self.softmax = nn.Softmax(dim=1)
		self.nn_human_x = nn.Parameter( torch.randn(NUM_PROTOTYPES, LATENT_SIZE), requires_grad=False)
		
	def __make_linear_weights(self):
		prototype_class_identity = torch.zeros(NUM_PROTOTYPES, NUM_CLASSES)
		num_prototypes_per_class = NUM_PROTOTYPES // NUM_CLASSES
		for j in range(NUM_PROTOTYPES):
			prototype_class_identity[j, j // num_prototypes_per_class] = 1
		positive_one_weights_locations = torch.t(prototype_class_identity)
		negative_one_weights_locations = 1 - positive_one_weights_locations
		incorrect_strength = 0.0
		correct_class_connection = 1
		incorrect_class_connection = incorrect_strength
		self.linear.weight.data.copy_(
			correct_class_connection * positive_one_weights_locations
			+ incorrect_class_connection * negative_one_weights_locations)
		
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
	
	def forward(self, x):
		
		# Let's pretend there's a variable like self.precomputed_protos that can be used for caching after training.
		latent_protos = None
		if self.prototypes is None:
			trans_nn_human_x = list()
			for i, t in enumerate(self.ts):
				trans_nn_human_x.append( t( torch.tensor(self.nn_human_x[i], dtype=torch.float32).view(1, -1)) )
			latent_protos = torch.cat(trans_nn_human_x, dim=0)   
		else:
			latent_protos = self.prototypes
			
		# Now we redo the logic that was in self.__transforms()
		p_acts = list()
		for i, t in enumerate(self.ts):
			action_prototype = latent_protos[i]
			p_acts.append( self.__proto_layer_l2( t(x), action_prototype).view(-1, 1) )
		p_acts = torch.cat(p_acts, axis=1)
		
		# And the final transformations:
		logits = self.linear(p_acts)                     
		final_outputs = self.__output_act_func(logits)   
		
		return final_outputs




final_accs = list()
final_rewards = list()

for _ in range(NUM_ITERATIONS):

	# Human defined Prototypes for interpretable model (these were gotten manually earlier)
	human_concepts = {'move_left':    [0.], 'move_right': [1.],
					 }
	human_concepts_list = np.array([l for l in human_concepts.values()])

	# Get prototypes with mean centres
	p_idxs = list()
	nn_human_x = list()

	for i in range(NUM_CLASSES):
		idxs = a_train == i
		temp_x = X_train[idxs]
		mean = temp_x.mean(axis=0)
		knn = KNeighborsClassifier().fit(temp_x, list(range(len(temp_x))))
		idx = knn.kneighbors(X=mean.reshape(1,-1), n_neighbors=1, return_distance=False)
		p_idxs.append(idx.item())
		nn_human_x.append( temp_x[idx.item()].tolist() )
	nn_human_x = np.array(nn_human_x)


	#### Training
	model = PWNet().eval()
	model.nn_human_x.data.copy_( torch.tensor(nn_human_x) )

	cce_loss = nn.CrossEntropyLoss()
	mse_loss = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, )
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
	best_acc = 0.
	model.train()
	loss_data = list()

	# Freeze Linear Layer to make more interpretable
	model.linear.weight.requires_grad = False


	for epoch in range(NUM_EPOCHS):
		running_loss = 0
		model.eval()
		current_acc = evaluate_loader(model, train_loader, cce_loss)[0]
		model.train()

		if current_acc > best_acc:
			torch.save(  model.state_dict(), 'weights/pw_net.pth'  )
			best_acc = current_acc

		for instances, labels in train_loader:
			optimizer.zero_grad()
			instances, labels = instances.to(DEVICE), labels.to(DEVICE)
			logits = model(instances)    
			loss = cce_loss(logits, labels)
			loss_data.append(loss.item())
			loss.backward()
			optimizer.step()
			running_loss += loss.item()

		print("Epoch:", epoch)
		print("Running Loss:", running_loss / len(train_loader))
		print("Acc.:", current_acc)
		print(" ")
		scheduler.step()


	#### Evaluate
	model = PWNet().eval()
	model.load_state_dict(torch.load('weights/pw_net.pth'))
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



