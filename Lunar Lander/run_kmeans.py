import gym
import torch 
import torch.nn as nn
import numpy as np      
import pandas as pd
import matplotlib.animation as animation
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch.optim as optim
import torch.nn.functional as F
import time
import json
import random

from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser
from os.path import join
from torch.distributions import Beta

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN, OPTICS

from random import sample
from tqdm import tqdm
from time import sleep

from collections import deque, Counter
from model import ActorCritic
from PIL import Image
 

MODEL_DIR = 'weights/k_means.pth'
NUM_CLASSES = 4
LATENT_SIZE = 128
PROTOTYPE_SIZE = 128
BATCH_SIZE = 32
DEVICE = 'cpu'
delay_ms = 0
NUM_PROTOTYPES = 4
NUM_SIMULATIONS = 30
NUM_ITERATIONS = 5


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


def evaluate_loader(model, loader, cce_loss):
    model.eval()
    total_correct = 0
    total_loss = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, labels = data
            
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)            
            logits, _ = model(imgs)
            loss = cce_loss(logits, labels)
            preds = torch.argmax(logits, dim=1)
            total_correct += sum(preds == labels).item()
            total += len(preds)
            total_loss += loss.item()  
    return (total_correct / total) * 100


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = toml.load(f)
    return config


def trans_human_concepts(model, nn_human_x):
    model.eval()
    trans_nn_human_x = list()
    for i, t in enumerate(model.ts):
        trans_nn_human_x.append( t( torch.tensor(nn_human_x[i], dtype=torch.float32).view(1, -1)) )
    model.train()
    return torch.cat(trans_nn_human_x, dim=0)


#### Start Collecting Data To Form Final Mean and Standard Error Results

data_rewards = list()
data_errors = list()

for _ in range(NUM_ITERATIONS):
    name='LunarLander_TWO.pth'
    env = gym.make('LunarLander-v2')
    policy = ActorCritic()
    policy.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    X_train = np.load('data/X_train.npy')
    a_train = np.load('data/a_train.npy')
    tensor_x = torch.Tensor(X_train)
    tensor_y = torch.tensor(a_train, dtype=torch.long)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)


    # Get prototypes with kmeans
    prototypes = list()
    for i in range(NUM_CLASSES):
        idxs = a_train == i
        temp_x = X_train[idxs]
        kmeans = KMeans(n_clusters=1, random_state=0).fit(temp_x)
        prototypes.append(kmeans.cluster_centers_.tolist()[0])
    prototypes = np.array(prototypes)


    #### Run simulations
    model = PW_Net(prototypes).eval()
    mse_loss = nn.MSELoss()
    cce_loss = nn.CrossEntropyLoss()
    all_acc = 0
    count = 0
    all_rewards = list()
    
    for i_episode in range(NUM_SIMULATIONS):
        state = env.reset()
        running_reward = 0
        
        for t in range(10000):
            bb_action, latent_x = policy(state)  # backbone latent x
            action = torch.argmax(  model(latent_x.view(1, -1))[0]  ).item()  # wrapper prediction
            state, reward, done, _ = env.step(action)
            running_reward += reward  
            all_acc += bb_action == action
            count += 1
            if done:
                break

    data_rewards.append(  running_reward  )
    data_errors.append(  all_acc / count  )

data_errors  = np.array(data_errors)
data_rewards = np.array(data_rewards)

print(data_errors)

print(" ")
print("===== Data Accuracy:")
print("Mean:", data_errors.mean())
print("Standard Error:", data_errors.std() / np.sqrt(NUM_ITERATIONS)  )
print(" ")
print("===== Data Reward:")
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt(NUM_ITERATIONS)  )






