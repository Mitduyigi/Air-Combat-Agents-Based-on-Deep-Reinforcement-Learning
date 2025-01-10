import wandb
import torch
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.scripts import main

if __name__ == '__main__':
    learning_rate = 1e-4
    num_episodes = 1000
    hidden_dim = [128, 64, 32]
    gamma = 0.99
    epsilon = 0.01
    tau = 0.01
    buffer_size = 1000000
    minimal_size = 10000
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    improvements = []

    for hd in hidden_dim:
        all_returns = []
        print("-"*80)
        print("Q网络隐藏层：", hd)
        wandb.init(
            project="LAG-SingleControl-hidden_dim2",
            config={
                "learning_rate": learning_rate,
                "num_episodes": num_episodes,
                "hidden_dim": hd,
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
        seeds = [0, 42, 115, 1673, 5258]
        for seed in seeds:
            print("-"*80)
            print("随机种子：", seed)
            returns = main(
                        learning_rate, num_episodes, hd, gamma, epsilon,
                        tau, buffer_size, minimal_size, batch_size, device, improvements, seed
                        )
            all_returns.append(returns[500:])  #统计在后半段episode时的episode return，来计算基本训练完成时的稳定性。
        '''all_returns_flattened = [r for returns in all_returns for r in returns]
        variance = np.var(all_returns_flattened)
        std = np.std(all_returns_flattened)
        mean = np.mean(all_returns_flattened)'''
        variance = np.mean([np.var(rt) for rt in all_returns])
        std = np.mean([np.std(rt) for rt in all_returns])
        mean = np.mean([np.mean(rt) for rt in all_returns])
        wandb.log({
            "variance": variance,
            "std": std,
            "mean": mean
        })

        wandb.finish()