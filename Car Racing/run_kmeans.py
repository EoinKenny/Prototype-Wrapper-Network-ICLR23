import gym
import torch 
import torch.nn as nn
import numpy as np      
import pickle
import toml

from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser
from os.path import join
from games.carracing import RacingNet, CarRacing
from ppo import PPO
from torch.distributions import Beta

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error

from random import sample
from tqdm import tqdm
from time import sleep


NUM_ITERATIONS = 5
MODEL_DIR = 'weights/kmeans_net.pth'
CONFIG_FILE = "config.toml"
NUM_CLASSES = 3
LATENT_SIZE = 256
PROTOTYPE_SIZE = 256
BATCH_SIZE = 32
NUM_EPOCHS = 50
DEVICE = 'cpu'
delay_ms = 0
NUM_PROTOTYPES = 4
SIMULATION_EPOCHS = 30


class KMeans_Net(nn.Module):

    def __init__(self, prototypes):
        super(KMeans_Net, self).__init__()
        self.prototypes = nn.Parameter(torch.tensor(prototypes, dtype=torch.float32), requires_grad=False)
        self.epsilon = 1e-5
        self.linear = nn.Linear(NUM_PROTOTYPES, NUM_CLASSES, bias=False) 
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU() 
        
    def __proto_layer_l2(self, x, p):
        output = list()
        b_size = x.shape[0]
        p = p.view(1, PROTOTYPE_SIZE).tile(b_size, 1).to(DEVICE) 
        c = x.view(b_size, PROTOTYPE_SIZE).to(DEVICE)      
        l2s = ( (c - p)**2 ).sum(axis=1).to(DEVICE) 
        act = torch.log( (l2s + 1. ) / (l2s + self.epsilon) ).to(DEVICE)  
        return act
    
    def __output_act_func(self, p_acts):        
        p_acts.T[0] = self.tanh(p_acts.T[0])  # steering between -1 -> +1
        p_acts.T[1] = self.relu(p_acts.T[1])  # acc > 0
        p_acts.T[2] = self.relu(p_acts.T[2])  # brake > 0
        return p_acts
    
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


def evaluate_loader(model, loader, loss):
    model.eval()
    total_error = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            current_loss = loss(logits, labels)
            total_error += current_loss.item()
            total += len(imgs)
    model.train()
    return total_error / total


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = toml.load(f)
    return config


data_rewards = list()
data_errors = list()

for _ in range(NUM_ITERATIONS):

    ## Load Pre-Trained Agent & Simulated Data
    cfg = load_config()
    env = CarRacing(frame_skip=0, frame_stack=4,)
    net = RacingNet(env.observation_space.shape, env.action_space.shape)
    ppo = PPO(
        env,
        net,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        gae_lambda=cfg["gae_lambda"],
        clip=cfg["clip"],
        value_coef=cfg["value_coef"],
        entropy_coef=cfg["entropy_coef"],
        epochs_per_step=cfg["epochs_per_step"],
        num_steps=cfg["num_steps"],
        horizon=cfg["horizon"],
        save_dir=cfg["save_dir"],
        save_interval=cfg["save_interval"],
    )
    ppo.load("weights/agent_weights.pt")

    with open('data/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/real_actions.pkl', 'rb') as f:
        real_actions = pickle.load(f)

    X_train = np.array([item for sublist in X_train for item in sublist])
    real_actions = np.array([item for sublist in real_actions for item in sublist])

    tensor_x = torch.Tensor(X_train)
    tensor_y = torch.tensor(real_actions, dtype=torch.float32)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader  = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    # Get the four prototypes
    kmeans = KMeans(n_clusters=4).fit(X_train)
    knn = KNeighborsClassifier().fit(X_train, list(range(len(X_train))))
    idxs = knn.kneighbors(X=kmeans.cluster_centers_, n_neighbors=1, return_distance=False)
    prototypes = X_train[idxs.flatten()]

    #### Train
    model = KMeans_Net(prototypes).eval()
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    best_error = float('inf')
    model.train()

    # Train linear layer to help k-Means baseline
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
            logits = model(instances)    
            loss = mse_loss(logits, labels)
            loss.backward()
            optimizer.step()
            
        scheduler.step()

    states, actions, rewards, log_probs, values, dones, X_train = [], [], [], [], [], [], []
    self_state = ppo._to_tensor(env.reset())

    # Wapper model with learned weights
    model = KMeans_Net(prototypes).eval()
    model.load_state_dict(torch.load(MODEL_DIR))

    reward_arr = []
    all_errors = list()

    for i in tqdm(range(SIMULATION_EPOCHS)):
        state = ppo._to_tensor(env.reset())
        count = 0
        rew = 0
        model.eval()

        for t in range(10000):
            # Get black box action
            value, alpha, beta, latent_x = ppo.net(state)
            value, alpha, beta = value.squeeze(0), alpha.squeeze(0), beta.squeeze(0)
            policy = Beta(alpha, beta)
            input_action = policy.mean.detach()
            _, _, _, _, bb_action = ppo.env.step(input_action.cpu().numpy())

            action = model(latent_x)
            all_errors.append(  mse_loss(bb_action, action[0]).detach().item()  )

            state, reward, done, _, _ = ppo.env.step(action[0].detach().numpy(), real_action=True)
            state = ppo._to_tensor(state)
            rew += reward
            count += 1

            if done:
                break

        reward_arr.append(rew)

    data_rewards.append(  sum(reward_arr) / SIMULATION_EPOCHS  )
    data_errors.append(  sum(all_errors) / SIMULATION_EPOCHS )

data_errors = np.array(data_errors)
data_rewards = np.array(data_rewards)

print(" ")
print("===== Data MAE:")
print("Mean:", data_errors.mean())
print("Standard Error:", data_errors.std() / np.sqrt(NUM_ITERATIONS)  )
print(" ")
print("===== Data Reward:")
print("Rewards:", data_rewards)
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt(NUM_ITERATIONS)  )




