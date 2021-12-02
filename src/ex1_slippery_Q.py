import gym
import numpy as np
from gym.envs.toy_text import FrozenLakeEnv
from frozenlake_utils import plot_results
from utils import env_step, env_reset

seed = 42
env = FrozenLakeEnv(is_slippery=False)
env.seed(seed)
np.random.seed(seed)

num_actions = env.action_space.n
num_observations = env.observation_space.n

Q = np.zeros((num_observations, num_actions))

alpha = 3e-1
eps = 1.0
gamma = 0.9
alpha_decay = 0.999
eps_decay = 0.999
max_train_iterations = 1000
max_test_iterations = 100
max_episode_length = 200


def policy(state, is_training):
    """ epsilon-greedy policy
    """

    if is_training and np.random.uniform(0, 1) < eps:  # choose actions deterministically during test phase
        return np.random.choice(num_actions)  # return a random action with probability epsilon
    else:
        return np.argmax(Q[state, :])  # otherwise return the action that maximizes Q


def train_step(state, action, reward, next_state, next_action, done):
    """ training using off-policy Q-learning algorithm
    """

    global alpha

    if done:  # terminal state reached
        Q[next_state, next_action] = 0

    Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.argmax(Q[next_state, :]) - Q[state, action])


def modify_reward(reward):
    # TODO: In some tasks, we will have to modify the reward.
    return reward


def run_episode(is_training):
    global eps
    episode_reward = 0
    state = env_reset(env, not is_training)
    action = policy(state, is_training)
    for t in range(max_episode_length):
        next_state, reward, done, _ = env_step(env, action, not is_training)
        reward = modify_reward(reward)
        episode_reward += reward
        next_action = policy(next_state, is_training)
        if is_training:
            train_step(state, action, reward, next_state, next_action, done)
        state, action = next_state, next_action
        if done:
            break
    return episode_reward


# Training phase
train_reward = []
for it in range(max_train_iterations):
    train_reward.append(run_episode(is_training=True))
    alpha *= alpha_decay
    eps *= eps_decay

# Test phase
test_reward = []
for it in range(max_test_iterations):
    test_reward.append(run_episode(is_training=False))
plot_results(train_reward, test_reward, Q, env)
