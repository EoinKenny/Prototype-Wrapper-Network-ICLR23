import gym
import numpy as np      
import pickle
import logging
import torch

from tqdm import tqdm
from time import sleep
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

from TD3 import TD3
from PIL import Image


### NB what depth is interpretable?
max_depth = 100

max_samples = 20000
env_step_limit = 200
simulation_lenght = 20000  # How many episodes per viper iteration
is_reweight = False
cross_val_splits = 10
NUM_ITERATIONS = 5
NUM_SIMULATIONS = 30
num_trees_check = 10

env_name = "BipedalWalker-v3"
random_seed = 0
n_episodes = 100
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
agent = TD3(lr, state_dim, action_dim, max_action)
agent.load_actor(directory, filename)

X_train = np.load('data/X_train.npy')
a_train = np.load('data/a_train.npy')
obs_train = np.load('data/obs_train.npy')

mse_loss = torch.nn.MSELoss()

def viper_sample(obss, labels, qs, max_pts=max_samples, is_reweight=is_reweight):
	
	"""
	Function taken from: https://github.com/obastani/viper
	observations
	latent x activations
	q values (only have softmax here)
	Max num observations
	Uniform or sampling
	"""
	
	# Step 1: Compute probabilities
	ps = np.max(qs, axis=1) - np.min(qs, axis=1)
	ps = ps / np.sum(ps)

	# Step 2: Sample points
	if is_reweight:
		# According to p(s)
		idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)), p=ps)
	else:
		# Uniformly (without replacement)
		idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)), replace=False)    

	# Step 3: Obtain sampled indices
	return obss[idx], labels[idx], qs[idx]


#### Step 2: Sample M trajectories
def sample_trajectories(policy, env):
	X_train = list()
	a_train = list()
	obs_train = list()

	for ep in range(n_episodes):
		ep_reward = 0
		state = env.reset()
		for t in range(max_timesteps):
			obs_train.append(state)
			A, x = policy.select_action(state)
			state, reward, done, _ = env.step(A)
			X_train.append(x)
			a_train.append(A)
			ep_reward += reward        
			if done:
				break

	X_train = np.array(X_train)
	a_train = np.array(a_train)
	obs_train = np.array(obs_train)
	return X_train, a_train, obs_train


def run_viper_simulation(agent, clf, env, knn_eval=False):
	reward_arr = list()
	for ep in range(simulation_lenght):
		ep_reward = 0
		state = env.reset()
		for t in range(max_timesteps):
			A = clf.predict(state.reshape(1, -1))
			state, reward, done, _ = env.step(A[0])
			ep_reward += reward        
			if done:
				reward_arr.append(ep_reward)
				break

	return sum(reward_arr) / len(reward_arr)


def evaluate_tree_viper(agent, env, x, y, latent_x):

	# Do cross validation for score: From VIPER paper
	dt  = DecisionTreeRegressor(random_state=0, max_depth=max_depth)
	df = MultiOutputRegressor(dt).fit(x, y)
	dt_cv_score  = cross_val_score(dt, x, y, cv=cross_val_splits, scoring='neg_mean_squared_error')
	dt.fit(x, y)

	# Also run simulations to get average reward
	avg_reward_dt = run_viper_simulation(agent, dt, env, knn_eval=False)
	print("DT CV Score:", dt_cv_score.mean())

	return dt_cv_score.mean(), avg_reward_dt, dt


data_rewards = list()
data_errors = list()

for _ in range(NUM_ITERATIONS):
	X_train = np.load('data/X_train.npy')
	a_train = np.load('data/a_train.npy')
	obs_train = np.load('data/obs_train.npy')
	logging.info("Initialising Dataset...")

	#### Step 1: Initialise dataset
	D = None
	A = None
	Q = None
	O = None

	# For picking best tree
	best_accs = 0
	best_rew = 0

	best_dt_classifier = None
	best_iteration = None

	best_acc = np.NINF
	best_rew = 0

	logging.info("Initialising Training...")
	## Start For Loop
	for iteration in tqdm(range(num_trees_check)):

		X_train, a_train, obs_train = sample_trajectories(agent, env)

		# Step 3: Aggregate dataset
		if iteration == 0:
			D = np.array(X_train)
			A = np.array(a_train)
			O = np.array(obs_train)
			
		else:
			D = np.append(D, X_train, axis=0)
			A = np.append(A, a_train, axis=0)
			O = np.append(O, obs_train, axis=0)

		# Step 4: Resample Dataset
		new_obs, new_a, new_x = viper_sample(O, A, D, max_pts=max_samples, is_reweight=is_reweight)

		# Step 6: Evaluate tree policy
		cv_score, avg_reward, clf = evaluate_tree_viper(agent, env, new_obs, new_a, new_x)

		logging.info("Current CV Score: " + str(cv_score))
		if cv_score > best_acc:
			best_acc = cv_score
			best_rew = avg_reward
			best_dt_classifier = clf
			best_iteration = iteration

	logging.info("Final Model on Iteration: " + str(best_iteration))
	logging.info("Best Accuracy CV: " + str(best_acc))
	logging.info("Best Reward: " + str(best_rew))

	filename = 'weights/viper_dt.sav'
	pickle.dump( best_dt_classifier, open(filename, 'wb') )

	# load the model from disk
	filename = 'weights/viper_dt.sav'
	viper_dt = pickle.load(open(filename, 'rb'))

	total_reward = list()
	all_errors = list()

	for i_episode in range(NUM_SIMULATIONS):
		ep_reward = 0
		ep_errors = 0
		state = env.reset()

		for t in range(max_timesteps):
			bb_action, x = agent.select_action(state)
			A = viper_dt.predict(state.reshape(1, -1))
			state, reward, done, _ = env.step(A[0])

			ep_reward += reward
			ep_errors += mse_loss( torch.tensor(bb_action), torch.tensor(A[0]) ).detach().item()

			if done:
				break
				
		total_reward.append( ep_reward )
		all_errors.append( ep_errors )
		ep_reward = 0
	env.close()  
	data_rewards.append( sum(total_reward) / NUM_SIMULATIONS )      
	data_errors.append( sum(all_errors) / NUM_SIMULATIONS )      

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


