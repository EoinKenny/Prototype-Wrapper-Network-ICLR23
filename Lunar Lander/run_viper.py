import gym
import numpy as np      
import pickle
import logging
import torch

from tqdm import tqdm
from time import sleep
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from model import ActorCritic


### NB what depth is interpretable?
max_depth = 100

max_samples = 20000
env_step_limit = 200
simulation_lenght = 30  # How many episodes per viper iteration
is_reweight = False
cross_val_splits = 10
NUM_ITERATIONS = 5
name='LunarLander_TWO.pth'
env = gym.make('LunarLander-v2')
agent = ActorCritic()
agent.load_state_dict(torch.load('./preTrained/{}'.format(name)))

X_train = np.load('data/X_train.npy')
a_train = np.load('data/a_train.npy')
obs_train = np.load('data/obs_train.npy')


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
	q_train = list()

	for i_episode in range(simulation_lenght):
		state = env.reset()
		running_reward = 0
		for t in range(10000):
			action, latent_x = policy(state)
			state, reward, done, _ = env.step(action)
			running_reward += reward
			X_train.append(latent_x.detach().tolist())
			a_train.append(action)
			obs_train.append(state.tolist()) 
			if done:
				break

	X_train = np.array(X_train)
	a_train = np.array(a_train)
	obs_train = np.array(obs_train)
	return X_train, a_train, obs_train


def run_viper_simulation(agent, clf, env, knn_eval=False):
	reward_arr = list()
	for i_episode in range(simulation_lenght):
		state = env.reset()
		running_reward = 0
		for t in range(10000):
			action, latent_x = agent(state)
			state, reward, done, _ = env.step(action)
			running_reward += reward	 
			if done:
				reward_arr.append(running_reward)
				break
	return sum(reward_arr) / len(reward_arr)


def evaluate_tree_viper(agent, env, x, y, latent_x):

	# Do cross validation for score: From VIPER paper
	knn = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
	dt  = DecisionTreeClassifier(random_state=0, max_depth=max_depth)
	dt_cv_score  = cross_val_score(dt, x, y, cv=cross_val_splits)
	dt.fit(x, y)

	# Also run simulations to get average reward: Ultimately more important
	avg_reward_dt = run_viper_simulation(agent, dt, env, knn_eval=False)
	print("DT CV Score:", dt_cv_score.mean())

	return dt_cv_score.mean(), avg_reward_dt, None, dt


data_rewards = list()
data_errors = list()

for _ in range(NUM_ITERATIONS):
	X_train = np.load('data/X_train.npy')
	a_train = np.load('data/a_train.npy')
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

	best_acc = 0
	best_rew = 0

	logging.info("Initialising Training...")
	## Start For Loop
	for iteration in tqdm(range(10)):
		X_train, a_train, obs_train = sample_trajectories(agent, env)

		# Step 3: Aggregate dataset
		if iteration == 0:
			D = np.array(X_train)
			A = np.array(a_train)
			O = np.array(obs_train)
			
		else:
			D = np.append(D, X_train, axis=0)
			A = np.append(A, a_train)
			O = np.append(O, obs_train, axis=0)

		# Step 4: Resample Dataset
		new_obs, new_a, new_x = viper_sample(O, A, D, max_pts=max_samples, is_reweight=is_reweight)

		# Step 6: Evaluate tree policy
		cv_score, avg_reward, avg_ep_count, clf = evaluate_tree_viper(agent, env, new_obs, new_a, new_x)

		logging.info("Current CV Score: " + str(cv_score))
		if cv_score > best_acc:
			best_acc = cv_score
			best_rew = avg_reward
			best_ep_count = avg_ep_count
			best_dt_classifier = clf
			best_iteration = iteration

	logging.info("Final Model on Iteration: " + str(best_iteration))
	logging.info("Best Accuracy CV: " + str(best_acc))
	logging.info("Best Reward: " + str(best_rew))
	logging.info("Best Episode Count: " + str(best_ep_count))

	filename = 'weights/viper_dt.sav'
	pickle.dump( best_dt_classifier, open(filename, 'wb') )

	NUM_SIMULATIONS = 30
	name='LunarLander_TWO.pth'
	env = gym.make('LunarLander-v2')

	# load the model from disk
	filename = 'weights/viper_dt.sav'
	viper_dt = pickle.load(open(filename, 'rb'))

	reward_arr = []
	for i_episode in range(NUM_SIMULATIONS):
		state = env.reset()
		running_reward = 0
		for t in range(10000):
			action = viper_dt.predict( [state] )
			state, reward, done, _ = env.step(  action.item()  )
			running_reward += reward                 
			if done:
				break
	env.close()

	data_rewards.append(  running_reward  )
	preds = viper_dt.predict(obs_train)
	data_errors.append(  accuracy_score(a_train, preds)  )

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











