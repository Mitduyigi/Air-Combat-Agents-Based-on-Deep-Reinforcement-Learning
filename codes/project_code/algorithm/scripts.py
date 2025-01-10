import torch
import numpy as np
import random
from tqdm import tqdm
import wandb
import time

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dqn import DQN
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from LAG.envs.JSBSim.envs.singlecontrol_env import SingleControlEnv


def main(lr, num_episodes, hidden_dim, gamma, epsilon,
        tau, buffer_size, minimal_size, batch_size, device, improvements, 
        seed, model_save_path=None, save_episode_place=None):
    env = SingleControlEnv("1/heading")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dims = list(env.action_space.nvec)

    agent = DQN(state_dim, hidden_dim, action_dims, lr, gamma, epsilon, tau, buffer_size, device, improvements)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset() #(12,)
                done = False
                while not done:
                    timestep = 0
                    actions = [agent.take_action(state)]
                    next_state, reward, done, _ = env.step(actions)
                    reward = reward * 10 / 3
                    agent.replay_buffer.add(state, actions, reward, next_state, done)
                    state = next_state
                    episode_return += reward * gamma ** timestep

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
                    timestep += 1
                return_list.append(episode_return)
                '''if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                    wandb.log({
                        "reward": np.mean(return_list[-10:]),
                        "episode": num_episodes / 10 * i + i_episode + 1,
                    })'''
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-1:])
                })
                wandb.log({
                    "reward": np.mean(return_list[-1:]),
                    "episode": num_episodes / 10 * i + i_episode + 1,
                })
                pbar.update(1)

                if model_save_path != None and save_episode_place != None and (num_episodes / 10 * i + i_episode + 1) in save_episode_place:
                    # 以.pt的格式保存整个模型
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    save_path = f"models/{model_save_path}_{num_episodes / 10 * i + i_episode + 1}_{timestamp}.pt"
                    save_dir = os.path.dirname(save_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(agent.q_net.state_dict(), save_path)
                    print(f"Model saved to {save_path}")

    return return_list

if __name__ == '__main__':
    learning_rate = 1e-4
    num_episodes = 800
    hidden_dim = 64
    gamma = 0.99
    epsilon = 0.01
    tau = 0.01
    buffer_size = 1000000
    minimal_size = 10000
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    improvements = []
    model_save_path = "train_best"
    seed = 42

    wandb.init(
        project="LAG-test",
        config={
            "learning_rate": learning_rate,
            "num_episodes": num_episodes,
            "hidden_dim": hidden_dim,
            "gamma": gamma,
            "epsilon": epsilon,
            "tau": tau,
            "buffer_size": buffer_size,
            "minimal_size": minimal_size,
            "batch_size": batch_size,
            "improvements": improvements
        },
        mode="offline",
    )

    main(
        learning_rate, num_episodes, hidden_dim, gamma, epsilon,
        tau, buffer_size, minimal_size, batch_size, 
        device, improvements, seed, model_save_path
    )

    wandb.finish()