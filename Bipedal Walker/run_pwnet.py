import gym
import numpy as np
import torch
import torch.nn as nn
import numpy as np      
import pandas as pd
import pickle

from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser
from os.path import join

from TD3 import TD3
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN, OPTICS

from random import sample
from tqdm import tqdm
from time import sleep
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error

from random import sample
from tqdm import tqdm
from time import sleep
from collections import Counter


SANITY_CHECK = False

NUM_CLASSES = 4
NUM_PROTOTYPES = 8
LATENT_SIZE = 300
BATCH_SIZE = 128
NUM_EPOCHS = 100
DEVICE = 'cpu'
PROTOTYPE_SIZE = 50
MAX_SAMPLES = 100000
MODEL_DIR = 'weights/pwnet.pth'
delay_ms = 0
SIMULATION_EPOCHS = 30
NUM_ITERATIONS = 5

env_name = "BipedalWalker-v3"
random_seed = 0
n_episodes = 30
lr = 0.002
max_timesteps = 2000
render = True
save_gif = False
filename = "TD3_{}_{}".format(env_name, random_seed)
filename += '_solved'
directory = "./preTrained/{}".format(env_name)
env = gym.make(env_name, hardcore=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
policy = TD3(lr, state_dim, action_dim, max_action)
policy.load_actor(directory, filename)


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
        self.tanh = nn.Tanh()
        self.nn_human_x = nn.Parameter( torch.randn(NUM_PROTOTYPES, LATENT_SIZE), requires_grad=False)
        
    def __make_linear_weights(self):
        custom_weight_matrix = torch.tensor([
                                             [ 1., 0., 0., 0.], 
                                             [ -1., 0., 0., 0.], 
                                             [ 0., 1., 0., 0.], 
                                             [ 0., -1., 0., 0.], 
                                             [ 0., 0., 1., 0.],
                                             [ 0., 0., -1., 0.], 
                                             [ 0., 0., 0., 1.], 
                                             [ 0., 0., 0., -1.], 
                                             ])
        self.linear.weight.data.copy_(custom_weight_matrix.T)   
        
    def __proto_layer_l2(self, x, p):
        output = list()
        b_size = x.shape[0]
        p = p.view(1, PROTOTYPE_SIZE).tile(b_size, 1).to(DEVICE) 
        c = x.view(b_size, PROTOTYPE_SIZE).to(DEVICE)      
        l2s = ( (c - p)**2 ).sum(axis=1).to(DEVICE) 
        act = torch.log( (l2s + 1. ) / (l2s + self.epsilon) ).to(DEVICE)  
        return act
    
    def __output_act_func(self, p_acts):        
        return self.tanh(p_acts)
    
    def forward(self, x):
        
        latent_protos = None
        if self.prototypes is None:
            trans_nn_human_x = list()
            for i, t in enumerate(self.ts):
                trans_nn_human_x.append( t( torch.tensor(self.nn_human_x[i], dtype=torch.float32).view(1, -1)) )
            latent_protos = torch.cat(trans_nn_human_x, dim=0)   
        else:
            latent_protos = self.prototypes
            
        p_acts = list()
        for i, t in enumerate(self.ts):
            action_prototype = latent_protos[i]
            p_acts.append( self.__proto_layer_l2( t(x), action_prototype).view(-1, 1) )
        p_acts = torch.cat(p_acts, axis=1)
        
        logits = self.linear(p_acts)                     
        final_outputs = self.__output_act_func(logits)   
        
        return final_outputs





def evaluate_loader(model, loader, mse_loss):
    model.eval()
    total_loss = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = mse_loss(outputs, labels)
            total += len(outputs)
            total_loss += loss.item()
    model.train()
    return total_loss / len(loader)


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = toml.load(f)
    return config


def proto_loss(model, nn_human_x, criterion):
    model.eval()
    target_x = trans_human_concepts(model, nn_human_x)
    loss = criterion(model.prototypes, target_x) 
    model.train()
    return loss



def trans_human_concepts(model, nn_human_x):
    
    model.eval()
    
    with torch.no_grad():
        trans_nn_human_x = list()
        for i, t in enumerate(model.ts):
            trans_nn_human_x.append( t( torch.tensor(nn_human_x[i], dtype=torch.float32).view(1, -1)) )
        
    model.train()

    return torch.cat(trans_nn_human_x, dim=0)



#### Start Collecting Data To Form Final Mean and Standard Error Results
data_rewards = list()
data_errors = list()

for _ in range(NUM_ITERATIONS):
    X_train = np.load('data/X_train.npy')
    a_train = np.load('data/a_train.npy')
    tensor_x = torch.Tensor(X_train)
    tensor_y = torch.tensor(a_train, dtype=torch.float32)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    # Get prototypes
    human_concepts = {'Hip1_Forward':  [1., 0., 0., 0.], 'Hip1_Back' :     [-1., 0., 0., 0.],
                      'Knee1_Forward': [0., 1., 0., 0.], 'Knee1_Back' :    [0., -1., 0., 0.],
                      'Hip2_Forward':  [0., 0., 1., 0.], 'Hip2_Back' :     [0., 0., -1., 0.],
                      'Knee2_Forward': [0., 0., 0., 1.], 'Knee2_Back' :    [0., 0., 0., -1.],
                     }
    human_concepts_list = np.array([l for l in human_concepts.values()])
    n_neighbours = 1
    knn = KNeighborsClassifier(algorithm='brute')
    knn.fit(a_train, list(range(len(a_train))))
    p_idxs = knn.kneighbors(X=human_concepts_list, n_neighbors=n_neighbours, return_distance=False)
    # nn_human_images = observations[p_idxs.flatten()]

    if SANITY_CHECK:
        p_idxs = np.random.randint(0, len(X_train), NUM_PROTOTYPES)

    nn_human_x = X_train[p_idxs.flatten()]
    nn_human_actions = a_train[p_idxs.flatten()]



    #### Training
    model = PWNet().eval()
    model.nn_human_x.data.copy_( torch.tensor(nn_human_x) )

    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    best_error = float('inf')
    model.train()

    loss_data = list()

    # Freeze Linear Layer to make more interpretable
    model.linear.weight.requires_grad = False


    for epoch in range(NUM_EPOCHS):
        
        running_loss = 0
            
        model.eval()
        train_error = evaluate_loader(model, train_loader, mse_loss)
        model.train()
        
        if train_error < best_error:
            torch.save(  model.state_dict(), MODEL_DIR  )
            best_error = train_error
        
        for instances, labels in train_loader:
            
            optimizer.zero_grad()
                    
            instances, labels = instances.to(DEVICE), labels.to(DEVICE)
                            
            logits = model(instances)    
            loss = mse_loss(logits, labels)
            loss_data.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
                    
        print("Epoch:", epoch)
        print("Running Error:", running_loss / len(train_loader))
        print("MAE:", train_error)
        print(" ")
        scheduler.step()

    states, actions, rewards, log_probs, values, dones, X_train = [], [], [], [], [], [], []



    # Wapper model with learned weights
    model = PWNet().eval()
    model.load_state_dict(torch.load(MODEL_DIR))

    # Projection
    print("MSE Eval:", evaluate_loader(model, train_loader, mse_loss))
    


    total_reward = list()
    all_errors = list()
    model.eval()

    for ep in range(SIMULATION_EPOCHS):
        ep_reward = 0
        ep_errors = 0
        state = env.reset()

        for t in range(max_timesteps):
            bb_action, x = policy.select_action(state)
            A = model( torch.tensor(x, dtype=torch.float32).view(1, -1) )
            state, reward, done, _ = env.step(A.detach().numpy()[0])

            ep_reward += reward
            ep_errors += mse_loss( torch.tensor(bb_action), A[0]).detach().item()

            if done:
                break
                
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        total_reward.append( ep_reward )
        all_errors.append( ep_errors )
        ep_reward = 0

    env.close()  

    data_rewards.append( sum(total_reward) / SIMULATION_EPOCHS )      
    data_errors.append( sum(all_errors) / SIMULATION_EPOCHS )      

data_errors = np.array(data_errors)
data_rewards = np.array(data_rewards)


print(" ")
print("===== MSE :")
print("Mean:", data_errors.mean())
print("Standard Error:", data_errors.std() / np.sqrt(NUM_ITERATIONS)  )
print(" ")
print("===== Reward:")
print("Rewards:", data_rewards)
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt(NUM_ITERATIONS)  )














