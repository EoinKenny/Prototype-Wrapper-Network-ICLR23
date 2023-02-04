import gym
import numpy as np      
import pickle
import logging

from tqdm import tqdm
from time import sleep
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from model import DQN_Agent


### NB what depth is interpretable?
max_depth = 5
# The Utility of Explainable AI in Ad Hoc Human-Machine Teaming: depth 2
# Interpretable and Personalized Apprenticeship Scheduling: Learning Interpretable Scheduling Policies from Heterogeneous User Demonstrations: depth 5
# An Evaluation of the Usefulness of Case-Based Explanation: depth 5 

max_samples = 20000
env_step_limit = 200
simulation_lenght = 100  # How many episodes per viper iteration
is_reweight = True
cross_val_splits = 10
logging.basicConfig(filename='logs/viper.log', encoding='utf-8', level=logging.DEBUG)


env = gym.make("CartPole-v1").unwrapped
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
exp_replay_size = 256
agent = DQN_Agent(seed=1423, layer_sizes=[input_dim, 64, output_dim], lr=1e-3, sync_freq=5, exp_replay_size=exp_replay_size)
agent.load_pretrained_model("weights/cartpole-dqn.pth")





def viper_sample(obss, labels, acts, qs, max_pts=max_samples, is_reweight=is_reweight):
	
	"""
	Function taken from: https://github.com/obastani/viper
	observations
	latent x activations
	q values
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
	return obss[idx], labels[idx], acts[idx], qs[idx]


#### Step 2: Sample M trajectories
def sample_trajectories(agent, env):
	X_train = list()
	a_train = list()
	q_train = list()
	obs_train = list()
	ep_count = list()
	reward_arr = []
	for i in range(simulation_lenght):
		obs, done, rew = env.reset(), False, 0
		count = 0
		while not done and count < env_step_limit:
			obs_train.append(obs.tolist())
			A, latent_x, q_values = agent.get_action(obs, env.action_space.n, epsilon=0)
			obs, reward, done, info, _ = env.step(A.item())
			X_train.append(latent_x.cpu().detach().tolist())
			a_train.append(A.cpu().detach().item())
			q_train.append(q_values.cpu().detach().tolist())
			rew += reward
			count += 1
		reward_arr.append(rew)
		ep_count.append(count)
	X_train   = np.array(X_train)
	a_train   = np.array(a_train)
	q_train   = np.array(q_train)
	obs_train = np.array(obs_train)
	return X_train, a_train, q_train, obs_train


def run_viper_simulation(agent, clf, env, knn_eval=False):
	reward_arr = list()
	ep_count = list()

	for i in range(simulation_lenght):
		obs, done, rew = env.reset(), False, 0
		count = 0

		# The observation (obs) now becomes the latent representation
		if knn_eval:
			_, obs, _ = agent.get_action(obs, env.action_space.n, epsilon=1)

		while not done and count < 200:
			A = clf.predict(obs.reshape(1, -1))
			obs, reward, done, info, _ = env.step(A.item())

			if knn_eval:
				_, obs, _ = agent.get_action(obs, env.action_space.n, epsilon=1)
			
			rew += reward
			count += 1
		reward_arr.append(rew)
		ep_count.append(count)

	return sum(reward_arr) / len(reward_arr), sum(ep_count) / len(ep_count)


def evaluate_tree_viper(agent, env, x, y, latent_x):

	# Do cross validation for score: From VIPER paper
	knn = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
	dt  = DecisionTreeClassifier(random_state=0, max_depth=max_depth)
	dt_cv_score  = cross_val_score(dt, x, y, cv=cross_val_splits)
	# knn_cv_score = cross_val_score(knn, latent_x, y, cv=cross_val_splits)  # fit k-nn with latent x

	dt.fit(x, y)
	# knn.fit(latent_x, y)

	# Also run simulations to get average reward: Ultimately more important
	avg_reward_dt, avg_ep_count_dt = run_viper_simulation(agent, dt, env, knn_eval=False)
	# avg_reward_knn, avg_ep_count_knn = run_viper_simulation(agent, knn, env, knn_eval=True)

	# print("DT and kNN CV Score:", dt_cv_score.mean(), knn_cv_score.mean())
	print("DT CV Score:", dt_cv_score.mean())

	return dt_cv_score.mean(), avg_reward_dt, avg_ep_count_dt, dt



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

	X_train, a_train, q_train, obs_train = sample_trajectories(agent, env)

	# Step 3: Aggregate dataset
	if iteration == 0:
		D = np.array(X_train)
		A = np.array(a_train)
		Q = np.array(q_train)
		O = np.array(obs_train)
		
	else:
		D = np.append(D, X_train, axis=0)
		A = np.append(A, a_train)
		Q = np.append(Q, q_train, axis=0)
		O = np.append(O, obs_train, axis=0)

	# Step 4: Resample Dataset
	new_obs, new_a, new_x, new_q = viper_sample(O, A, D, Q, max_pts=max_samples, is_reweight=is_reweight)

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

# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))




