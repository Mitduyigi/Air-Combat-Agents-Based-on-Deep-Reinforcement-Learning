import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

from dqn import DQN

import sys
import os
sys.path.append(os.path.abspath("/Users/mitduyigi/Desktop/project"))

from LAG.envs.JSBSim.envs.singlecontrol_env import SingleControlEnv


def main(lr, num_episodes, hidden_dim, gamma, epsilon,
        target_update, buffer_size, minimal_size, batch_size, device, improvements):
    env = SingleControlEnv("1/heading")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env.seed(0)

    state_dim = env.observation_space.shape[0]
    action_dims = list(env.action_space.nvec)

    agent = DQN(state_dim, hidden_dim, action_dims, lr, gamma, epsilon, target_update, buffer_size, device, improvements)

    return_list = []
    for i in range(10):
        for i_episode in tqdm(range(int(num_episodes / 10))):
            episode_return = 0
            state = env.reset() #(12,)
            done = False
            while not done:
                actions = agent.take_action(state) #1152063
                actions = [agent.joint_to_indiv(actions)] #[[4, 14, 29, 16]]
                next_state, reward, done, _ = env.step(actions)
                agent.replay_buffer.add(state, actions, reward, next_state, done)
                state = next_state
                episode_return += reward

                if agent.replay_buffer.size() > minimal_size:
                    if "PrioritizedDQN" in improvements:
                        b_s, b_a, b_r, b_ns, b_d, indices, weights = agent.replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d,
                            "indices": indices,
                            "weights": weights
                        }
                        agent.update(transition_dict)
                    else:
                        b_s, b_a, b_r, b_ns, b_d = agent.replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
            return_list.append(episode_return)

if __name__ == '__main__':
    lr = 1e-3
    num_episodes = 100000
    hidden_dim = 64 
    gamma = 0.99
    epsilon = 0.01
    target_update = 10
    buffer_size = 1000000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    improvements = []

    main(
        lr, num_episodes, hidden_dim, gamma, epsilon,
        target_update, buffer_size, minimal_size, batch_size, device, improvements
    )
