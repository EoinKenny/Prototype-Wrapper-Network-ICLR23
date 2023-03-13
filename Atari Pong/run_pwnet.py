import gym
import torch 
import torch.nn as nn
import numpy as np      
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

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans, DBSCAN, OPTICS

from random import sample
from tqdm import tqdm
from time import sleep

from collections import deque


MODEL_DIR = 'weights/pw_net.pth'
NUM_CLASSES = 6
LATENT_SIZE = 1536
PROTOTYPE_SIZE = 50
BATCH_SIZE = 32
NUM_EPOCHS = 10
DEVICE = 'cpu'
delay_ms = 0
NUM_PROTOTYPES = 6
SIMULATION_EPOCHS = 30
NUM_ITERATIONS = 3



ENVIRONMENT = "PongDeterministic-v4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODELS = False  # Save models to file so you can test later
MODEL_PATH = "./models/pong-cnn-"  # Models path for saving or loading
SAVE_MODEL_INTERVAL = 10  # Save models at every X epoch
TRAIN_MODEL = False  # Train model while playing (Make it False when testing a model)
LOAD_MODEL_FROM_FILE = True  # Load model from file
LOAD_FILE_EPISODE = 900  # Load Xth episode from file
BATCH_SIZE = 64  # Minibatch size that select randomly from mem for train nets
MAX_EPISODE = 100000  # Max episode
MAX_STEP = 100000  # Max step size for one episode
NUM_EPISODES = 3
MAX_MEMORY_LEN = 50000  # Max memory len
MIN_MEMORY_LEN = 40000  # Min memory len before start train
GAMMA = 0.97  # Discount rate
ALPHA = 0.00025  # Learning rate
EPSILON_DECAY = 0.99  # Epsilon decay rate by step


class DuelCNN(nn.Module):
    """
    CNN with Duel Algo. https://arxiv.org/abs/1511.06581
    """

    def __init__(self, h, w, output_size):
        super(DuelCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4,  out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        convw, convh = self.conv2d_size_calc(w, h, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=3, stride=1)

        linear_input_size = convw * convh * 64  # Last conv layer's out sizes

        # Action layer
        self.Alinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Alrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Alinear2 = nn.Linear(in_features=128, out_features=output_size)

        # State Value layer
        self.Vlinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Vlrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Vlinear2 = nn.Linear(in_features=128, out_features=1)  # Only 1 node

    def conv2d_size_calc(self, w, h, kernel_size=5, stride=2):
        """
        Calcs conv layers output image sizes
        """
        next_w = (w - (kernel_size - 1) - 1) // stride + 1
        next_h = (h - (kernel_size - 1) - 1) // stride + 1
        return next_w, next_h

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten every batch

        Ax = self.Alrelu(self.Alinear1(x))
        Ax = self.Alinear2(Ax)  # No activation on last layer

        Vx = self.Vlrelu(self.Vlinear1(x))
        Vx = self.Vlinear2(Vx)  # No activation on last layer

        q = Vx + (Ax - Ax.mean())

        return q, x


class Agent:
    def __init__(self, environment):
        """
        Hyperparameters definition for Agent
        """

        # State size for breakout env. SS images (210, 160, 3). Used as input size in network
        self.state_size_h = environment.observation_space.shape[0]
        self.state_size_w = environment.observation_space.shape[1]
        self.state_size_c = environment.observation_space.shape[2]

        # Activation size for breakout env. Used as output size in network
        self.action_size = environment.action_space.n

        # Image pre process params
        self.target_h = 80  # Height after process
        self.target_w = 64  # Widht after process

        self.crop_dim = [20, self.state_size_h, 0, self.state_size_w]  # Cut 20 px from top to get rid of the score table

        # Trust rate to our experiences
        self.gamma = GAMMA  # Discount coef for future predictions
        self.alpha = ALPHA  # Learning Rate

        # After many experinces epsilon will be 0.05
        # So we will do less Explore more Exploit
        self.epsilon = 0  # Explore or Exploit
        self.epsilon_decay = EPSILON_DECAY  # Adaptive Epsilon Decay Rate
        self.epsilon_minimum = 0.05  # Minimum for Explore

        # Deque holds replay mem.
        self.memory = deque(maxlen=MAX_MEMORY_LEN)

        # Create two model for DDQN algorithm
        self.online_model = DuelCNN(h=self.target_h, w=self.target_w, output_size=self.action_size).to(DEVICE)
        self.target_model = DuelCNN(h=self.target_h, w=self.target_w, output_size=self.action_size).to(DEVICE)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

        # Adam used as optimizer
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.alpha)

    def preProcess(self, image):
        """
        Process image crop resize, grayscale and normalize the images
        """
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To grayscale
        frame = frame[self.crop_dim[0]:self.crop_dim[1], self.crop_dim[2]:self.crop_dim[3]]  # Cut 20 px from top
        frame = cv2.resize(frame, (self.target_w, self.target_h))  # Resize
        frame = frame.reshape(self.target_w, self.target_h) / 255  # Normalize

        return frame

    def act(self, state):
        """
        Get state and do action
        Two option can be selectedd if explore select random action
        if exploit ask nnet for action
        """

        act_protocol = 'Explore' if random.uniform(0, 1) <= self.epsilon else 'Exploit'

        if act_protocol == 'Explore':
            action = random.randrange(self.action_size)
            state = torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0)
            q_values, x = self.online_model.forward(state)  # (1, action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0)
                q_values, x = self.online_model.forward(state)  # (1, action_size)
                action = torch.argmax(q_values).item()  # Returns the indices of the maximum value of all elements

        return action, x

    def train(self):
        """
        Train neural nets with replay memory
        returns loss and max_q val predicted from online_net
        """
        if len(agent.memory) < MIN_MEMORY_LEN:
            loss, max_q = [0, 0]
            return loss, max_q
        # We get out minibatch and turn it to numpy array
        state, action, reward, next_state, done = zip(*random.sample(self.memory, BATCH_SIZE))

        # Concat batches in one array
        # (np.arr, np.arr) ==> np.BIGarr
        state = np.concatenate(state)
        next_state = np.concatenate(next_state)

        # Convert them to tensors
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
        action = torch.tensor(action, dtype=torch.long, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)
        done = torch.tensor(done, dtype=torch.float, device=DEVICE)

        # Make predictions
        state_q_values = self.online_model(state)
        next_states_q_values = self.online_model(next_state)
        next_states_target_q_values = self.target_model(next_state)

        # Find selected action's q_value
        selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # Get indice of the max value of next_states_q_values
        # Use that indice to get a q_value from next_states_target_q_values
        # We use greedy for policy So it called off-policy
        next_states_target_q_value = next_states_target_q_values.gather(1, next_states_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        # Use Bellman function to find expected q value
        expected_q_value = reward + self.gamma * next_states_target_q_value * (1 - done)

        # Calc loss with expected_q_value and q_value
        loss = (selected_q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, torch.max(state_q_values).item()

    def storeResults(self, state, action, reward, nextState, done):
        """
        Store every result to memory
        """
        self.memory.append([state[None, :], action, reward, nextState[None, :], done])

    def adaptiveEpsilon(self):
        """
        Adaptive Epsilon means every step
        we decrease the epsilon so we do less Explore
        """
        if self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay



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


#### Start Collecting Data To Form Final Mean and Standard Error Results
data_rewards = list()
data_errors = list()

for _ in range(NUM_ITERATIONS):

    environment = gym.make(ENVIRONMENT) # , render_mode='human')  # Get env
    environment.seed(0)
    agent = Agent(environment)  # Create Agent
    if LOAD_MODEL_FROM_FILE:
        agent.online_model.load_state_dict(torch.load(MODEL_PATH+str(LOAD_FILE_EPISODE)+".pkl", map_location=torch.device('cpu')))
        with open(MODEL_PATH+str(LOAD_FILE_EPISODE)+'.json') as outfile:
            param = json.load(outfile)
            agent.epsilon = param.get('epsilon')
        startEpisode = LOAD_FILE_EPISODE + 1
    else:
        startEpisode = 1
    last_100_ep_reward = deque(maxlen=100)  # Last 100 episode rewards
    total_step = 1  # Cumulkative sum of all steps in episodes


    with open('data/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/a_train.pkl', 'rb') as f:
        a_train = pickle.load(f)

    X_train = np.array(X_train)
    a_train = np.array(a_train)
    tensor_x = torch.Tensor(X_train)
    tensor_y = torch.tensor(a_train, dtype=torch.long)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    # Human defined Prototypes for interpretable model (these were gotten manually earlier)
    human_concepts = {'stay1':    [0.], 'stay2' :      [1.],
                      'move_up1': [2.], 'move_down1' : [3.],
                      'move_up2': [4.], 'move_down2' : [5.],
                     }
    human_concepts_list = np.array([l for l in human_concepts.values()])

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

    states, actions, rewards, log_probs, values, dones, X_train = [], [], [], [], [], [], []



    # Wapper model with learned weights
    model = PWNet().eval()
    model.load_state_dict(torch.load(MODEL_DIR))

    # Projection
    print("Final Acc.:", evaluate_loader(model, train_loader, cce_loss))


    all_rewards = list()
    all_errors = list()
    for episode in range(SIMULATION_EPOCHS):

        startTime = time.time()  # Keep time
        state = environment.reset()  # Reset env

        state = agent.preProcess(state)  # Process image

        # Stack state . Every state contains 4 time contionusly frames
        # We stack frames like 4 channel image
        state = np.stack((state, state, state, state))

        total_max_q_val = 0  # Total max q vals
        total_reward = 0     # Total reward for each episode
        total_loss = 0       # Total loss for each episode
        total_error = list()

        for step in range(MAX_STEP):

            # Select and perform an action
            agent_action, latent_x = agent.act(state)  # Act
            action = torch.argmax(model(latent_x)).item()

            # print(agent_action, action)

            # Normally the randomness is the number on the right (.049...)
            # But as PW-Net is trained on the data from the original model which was already random
            # we lower the randomness here for a fairer comparison.
            # PW-Net here is trained on ~5% random data, plus 0.025 randomness
            if np.random.random_sample() < .025:   #  .04953625663766238:
                action = np.random.randint(0, 5)

            next_state, reward, done, info = environment.step(action)  # Observe

            next_state = agent.preProcess(next_state)  # Process image

            # Stack state . Every state contains 4 time contionusly frames
            # We stack frames like 4 channel image
            next_state = np.stack((next_state, state[0], state[1], state[2]))

            # Store the transition in memory
            agent.storeResults(state, action, reward, next_state, done)  # Store to mem

            # Move to the next state
            state = next_state  # Update state

            total_reward += reward
            total_error.append( agent_action == action )

            if done:
                all_rewards.append(total_reward)
                all_errors.append( sum(total_error) / len(total_error ) )
                break

    data_rewards.append(  sum(all_rewards) / SIMULATION_EPOCHS  )
    data_errors.append(  sum(all_errors) / SIMULATION_EPOCHS  )
    print(" ")
    print("==========================")
    print("Rewards:", data_rewards)
    print("Accuracy:", data_errors)
    print("==========================")
    print(" ")

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



