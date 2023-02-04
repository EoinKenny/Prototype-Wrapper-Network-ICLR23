import gym
import numpy as np
import torch
import torch.nn as nn
import gym
import torch 
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
from sklearn.metrics import mean_absolute_error


MODEL_DIR = 'weights/kmeans_net.pth'
NUM_CLASSES = 4
LATENT_SIZE = 300
PROTOTYPE_SIZE = 300
BATCH_SIZE = 32
NUM_EPOCHS = 50
DEVICE = 'cpu'
delay_ms = 0
NUM_PROTOTYPES = 4
SIMULATION_EPOCHS = 30
NUM_ITERATIONS = 5

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
        return final_outputs, None


def evaluate_loader(model, loader, loss):
    model.eval()
    total_error = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits, _ = model(imgs)
            current_loss = loss(logits, labels)
            total_error += current_loss.item()
            total += len(imgs)
    model.train()
    return total_error / total


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = toml.load(f)
    return config





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

    # Get prototypes with kmeans
    kmeans = KMeans(n_clusters=NUM_PROTOTYPES).fit(X_train)
    knn = KNeighborsClassifier().fit(X_train, list(range(len(X_train))))
    idxs = knn.kneighbors(X=kmeans.cluster_centers_, n_neighbors=1, return_distance=False)
    prototypes = X_train[idxs.flatten()]






    #### Train
    model = PW_Net(prototypes).eval()
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    best_error = float('inf')
    model.train()

    # Train linear layer to help baseline
    model.linear.weight.requires_grad = True

    for epoch in range(NUM_EPOCHS):
        
        model.eval()
        train_error = evaluate_loader(model, train_loader, mse_loss)
        model.train()
        
        if train_error < best_error:
            torch.save(model.state_dict(), MODEL_DIR)
            best_error = train_error
        
        for instances, labels in train_loader:
            optimizer.zero_grad()     
            instances, labels = instances.to(DEVICE), labels.to(DEVICE)
            logits, _ = model(instances)    
            loss = mse_loss(logits, labels)
            loss.backward()
            optimizer.step()
            
        scheduler.step()


    #### Run Test
    model = PW_Net(prototypes).eval()
    model.load_state_dict(torch.load(MODEL_DIR))

    total_reward = list()
    all_errors = list()

    for ep in range(SIMULATION_EPOCHS):
        ep_reward = 0
        ep_errors = 0
        state = env.reset()
        
        for t in range(max_timesteps):
            bb_action, x = policy.select_action(state)
            A, _ = model( torch.tensor(x, dtype=torch.float32).view(1, -1) )
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
print("===== Data Accuracy:")
print("Mean:", data_errors.mean())
print("Standard Error:", data_errors.std() / np.sqrt(NUM_ITERATIONS)  )
print(" ")
print("===== Data Reward:")
print("Rewards:", data_rewards)
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt(NUM_ITERATIONS)  )














