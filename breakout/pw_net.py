# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License

from __future__ import print_function
import torch, os, gym, time, glob, argparse, sys
import numpy as np
from scipy.signal import lfilter

import cv2 # preserves single-pixel info _unlike_ img = img[::2,::2]

import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import pickle

import gym
import torch 
import torch.nn as nn
import numpy as np      
import pandas as pd
import pickle
import toml
import cv2
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import json
import random

from collections import Counter
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser
from os.path import join
from torch.distributions import Beta
from IPython.display import HTML

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans, DBSCAN, OPTICS

from random import sample
from tqdm import tqdm
from time import sleep

from collections import deque



NUM_SIMULATIONS = 30
MODEL_DIR = 'weights/pw_net.pth'
NUM_CLASSES = None
LATENT_SIZE = 256
PROTOTYPE_SIZE = 50
BATCH_SIZE = 32
NUM_EPOCHS = 50
DEVICE = 'cpu'
delay_ms = 0
NUM_PROTOTYPES = None
NUM_ITERATIONS = 5



os.environ['OMP_NUM_THREADS'] = '1'

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Breakout-v4', type=str, help='gym environment')
    parser.add_argument('--processes', default=1, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=True, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    return parser.parse_args()

discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # discounted rewards one liner
prepro = lambda img: cv2.resize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) ; f=open(args.save_dir+'log.txt',mode) ; f.write(s+'\n') ; f.close()

class NNPolicy(nn.Module): # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        hx = self.gru(x.view(-1, 32 * 5 * 5), (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step

class SharedAdam(torch.optim.Adam): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)

def cost_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, actions.clone().detach().view(-1,1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()
    
    # l2 loss over value estimator
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum()

    entropy_loss = (-logps * torch.exp(logps)).sum() # entropy definition, for entropy regularization
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss



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





def run(shared_model, shared_optimizer, args, info, pw_net_model):

    all_rewards = list()
    all_accs = list()

    for rand_seed, simulation in enumerate(range(NUM_SIMULATIONS)): 
        env = gym.make(args.env)  #  , render_mode='human') # make a local (unshared) environment
        env.seed(rand_seed) ; torch.manual_seed(rand_seed) # seed everything
        model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions) # a local/unshared model
        state = torch.tensor(prepro(env.reset())) # get first state
        start_time = last_disp_time = time.time()
        episode_length, epr, eploss, done  = 0, 0, 0, False # bookkeeping

        model.load_state_dict(shared_model.state_dict()) # sync with shared model
        hx = torch.zeros(1, 256)  #    if done else hx.detach()  # rnn activation vector
        values, logps, actions, rewards = [], [], [], [] # save values for computing gradientss

        max_episodes = 1000

        while episode_length < max_episodes:
            episode_length += 1
            value, logit, hx = model((state.view(1,1,80,80), hx))
            logp = F.log_softmax(logit, dim=-1)

            # Black-box prediction (just sanity check)
            AgentAction = torch.argmax(logit).item()  # logp.max(1)[1].data if args.test else

            # Wrapper prediction
            action = torch.argmax(pw_net_model(hx)).item()

            state, reward, done, _ = env.step(action)

            state = torch.tensor(prepro(state))
            epr += reward
            reward = np.clip(reward, -1, 1)  # reward

            all_accs.append( AgentAction == action )

            # print(AgentAction, action)


            if episode_length == max_episodes: # maybe print info.
                all_rewards.append( epr )
                print("Episode Length:", episode_length)
                print("Reward:", epr)
                print(" ")
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(prepro(env.reset()))
                done = True
                break


        next_value = torch.zeros(1,1) if done else model((state.unsqueeze(0), hx))[0]
        values.append(next_value.detach())

    all_rewards = np.array(all_rewards)
    all_accs = np.array(all_accs)

    return all_rewards.mean(), all_accs.mean()




def train_pwnet():

    with open('data/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/a_train.pkl', 'rb') as f:
        a_train = pickle.load(f)

    NUM_CLASSES = len(Counter(a_train).keys())
    NUM_PROTOTYPES = NUM_CLASSES

    X_train = np.array(X_train)
    a_train = np.array(a_train)
    tensor_x = torch.Tensor(X_train)
    tensor_y = torch.tensor(a_train, dtype=torch.long)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

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
    nn_human_x = np.array(nn_human_x)


    #### Training
    model = PWNet().eval()
    model.nn_human_x.data.copy_( torch.tensor(nn_human_x) )

    cce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
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


    # Wapper model with learned weights
    model = PWNet().eval()
    model.load_state_dict(torch.load(MODEL_DIR))

    return model.eval()



if __name__ == "__main__":



    with open('data/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/a_train.pkl', 'rb') as f:
        a_train = pickle.load(f)

    NUM_CLASSES = len(Counter(a_train).keys())
    NUM_PROTOTYPES = NUM_CLASSES

    

    args = get_args()
    args.save_dir = '{}/'.format(args.env.lower()) # keep the directory structure simple
    if args.render:  args.processes = 1 ; args.test = True # render mode -> test mode w one process
    args.lr = 0 # don't train in render mode
    args.num_actions = gym.make(args.env).action_space.n # get the action space of this game
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None # make dir to save models etc.

    torch.manual_seed(args.seed)
    shared_model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions).share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    info['frames'] += shared_model.try_load(args.save_dir) * 1e6
    if int(info['frames'].item()) == 0: printlog(args,'', end='', mode='w') # clear log file

    

    data_errors = list()
    data_rewards = list()

    for i in range(NUM_ITERATIONS):

        pw_net_model = train_pwnet()

        iter_reward, iter_acc = run(shared_model, shared_optimizer, args, info, pw_net_model)
        data_errors.append(iter_acc)
        data_rewards.append(iter_reward)
    

    data_errors = np.array(data_errors)
    data_rewards = np.array(data_rewards)

    print("Data Errors:", data_errors)
    print("Data Rewards:", data_rewards)

    print(" ")
    print("===== Data Accuracy:")
    print("Mean:", data_errors.mean())
    print("Standard Error:", data_errors.std() / np.sqrt( NUM_ITERATIONS )  )
    print(" ")
    print("===== Data Reward:")
    print("Mean:", data_rewards.mean())
    print("Standard Error:", data_rewards.std() / np.sqrt( NUM_ITERATIONS )  )



