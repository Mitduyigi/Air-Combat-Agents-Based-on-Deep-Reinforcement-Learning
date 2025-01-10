import wandb
import torch
import numpy as np
import random
from tqdm import tqdm
import time

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from LAG.envs.JSBSim.envs import SingleCombatEnv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.dqn import DQN

def train_baseline(lr, num_episodes, hidden_dim, gamma, epsilon,
        tau, buffer_size, minimal_size, batch_size, device, improvements, 
        seed, model_save_path=None, save_episode_place=None):
    env = SingleCombatEnv(config_name='1v1/NoWeapon/Selfplay')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dims = list(env.action_space.nvec)

    agent0 = DQN(state_dim, hidden_dim, action_dims, lr, gamma, epsilon, tau, buffer_size, device, improvements)
    agent1 = DQN(state_dim, hidden_dim, action_dims, lr, gamma, epsilon, tau, buffer_size, device, improvements)

    return_list = {"agent0": [], "agent1": []}
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = [0, 0]
                state = env.reset()      #Selfplay (2,15)           #vsBaseline(1, 15)
                env.render()

                while True:
                    actions = [agent0.take_action(state[0]), agent1.take_action(state[1])]
                    next_state, reward, done, _ = env.step(actions)
                    env.render()
                    for num_agent in range(len([agent0, agent1])):
                        agent = [agent0, agent1][num_agent]
                        agent.replay_buffer.add(state[num_agent:(num_agent+1), :], [actions[num_agent]], [reward[num_agent]], next_state[num_agent:(num_agent+1), :], [done[num_agent]])
                        state = next_state
                        episode_return[num_agent] += reward[num_agent]

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
                    if np.array(done).all():
                        break
                return_list["agent0"].append(episode_return[0])
                return_list["agent1"].append(episode_return[1])
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'agent0 return':
                        '%.3f' % np.mean(return_list["agent0"][-10:]),
                        'agent1 return':
                        '%.3f' % np.mean(return_list["agent1"][-10:])
                    })
                    wandb.log({
                            "agent0 reward": np.mean(return_list["agent0"][-10:]),
                            "agent1 reward": np.mean(return_list["agent1"][-10:]),
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

def main(learning_rate, num_episodes, hidden_dim, gamma, epsilon,
        tau, buffer_size, minimal_size, batch_size, 
        device, improvements, seed, model_save_path, save_episode_place):
    later_half_returns = []
    wandb.init(
        project="LAG-1v1-NoWeapon-vsbaseline",
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
    seed = 42
    returns = train_baseline(
            learning_rate, num_episodes, hidden_dim, gamma, epsilon,
            tau, buffer_size, minimal_size, batch_size, 
            device, improvements, seed, model_save_path, save_episode_place
        )
    later_half_returns = {"agent0": returns["agent0"][500:], "agent1": returns["agent1"][500:]}  #统计在后半段episode时的episode return，来计算基本训练完成时的稳定性。
        
    variance0 = np.var(later_half_returns["agent0"])
    std0 = np.std(later_half_returns["agent0"])
    mean0 = np.mean(later_half_returns["agent0"])
    variance1 = np.var(later_half_returns["agent1"])
    std1 = np.std(later_half_returns["agent1"])
    mean1 = np.mean(later_half_returns["agent1"])
    wandb.log({
        "agent0 variance": variance0,
        "agent0 std": std0,
        "agent0 mean": mean0,
        "agent1 variance": variance1,
        "agent1 std": std1,
        "agent1 mean": mean1
    })

    wandb.finish()

if __name__ == '__main__':
    learning_rate = 1e-4
    num_episodes = 1000
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
    save_episode_place = [700, 800, 900, 1000]

    main(
        learning_rate, num_episodes, hidden_dim, gamma, epsilon,
        tau, buffer_size, minimal_size, batch_size, 
        device, improvements, seed, model_save_path, save_episode_place
    )