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


SANITY_CHECK = False

MODEL_DIR = 'weights/pwnet.pth'
NUM_CLASSES = 4
LATENT_SIZE = 128
PROTOTYPE_SIZE = 50
BATCH_SIZE = 32
NUM_EPOCHS = 100
DEVICE = 'cpu'
delay_ms = 0
NUM_PROTOTYPES = 4
NUM_SIMULATIONS = 30
NUM_ITERATIONS = 5


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


def evaluate_loader(model, loader, cce_loss):
    model.eval()
    total_correct = 0
    total_loss = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)            
            logits = model(imgs)
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


def proto_loss(model, nn_human_x, criterion):
    model.eval()
    target_x = trans_human_concepts(model, nn_human_x)
    loss = criterion(model.prototypes, target_x) 
    model.train()
    return loss


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

    human_concepts = {'nothing': [0.], 'left' : [1.], 'main': [2.], 'right' : [3.]}
    human_concepts_list = np.array([l for l in human_concepts.values()])
    n_neighbours = 1



    # Get prototypes with means centres
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
    p_idxs = np.array(p_idxs)
    nn_human_x = np.array(nn_human_x)


    if SANITY_CHECK:
        p_idxs = np.random.randint(0, len(X_train), NUM_PROTOTYPES)
        nn_human_x = X_train[p_idxs.flatten()]





    #### Training
    model = PWNet().eval()
    model.nn_human_x.data.copy_( torch.tensor(nn_human_x) )

    cce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    best_acc = 0.
    model.train()

    loss_data = list()

    # Freeze Linear Layer to make more interpretable
    model.linear.weight.requires_grad = False


    for epoch in range(NUM_EPOCHS):
        
        running_loss = 0
            
        model.eval()
        current_acc = evaluate_loader(model, train_loader, cce_loss)
        model.train()
        
        if current_acc > best_acc:
            torch.save(  model.state_dict(), MODEL_DIR  )
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

    states, actions, rewards, log_probs, values, dones, X_train = [], [], [], [], [], [], []

    # Wapper model with learned weights
    model = PWNet().eval()
    model.load_state_dict(torch.load(MODEL_DIR))

    # Projection
    print("Final Acc.:", evaluate_loader(model, train_loader, cce_loss))


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
        print("Reward:", running_reward)
    data_errors.append(  all_acc / count  )
    print("Avg Reward:", sum(data_rewards) / len(data_rewards))
    print("Avg Acc:", sum(data_errors) / len(data_errors))

data_errors = np.array(data_errors)
data_rewards = np.array(data_rewards)

print(" ")
print("===== Data Accuracy:")
print("Mean:", data_errors.mean())
print("Standard Error:", data_errors.std() / np.sqrt(NUM_ITERATIONS)  )
print(" ")
print("===== Data Reward:")
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt(NUM_ITERATIONS)  )






