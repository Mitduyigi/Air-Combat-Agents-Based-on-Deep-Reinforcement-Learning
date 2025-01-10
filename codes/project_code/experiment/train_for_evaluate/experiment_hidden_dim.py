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
    model_save_path = "hidden_dim"
    save_episode_place = [970, 980, 990, 1000] 

    for hd in hidden_dim:
        model_save_path = f"hidden_dim_{hd}"
        all_returns = []
        print("-"*80)
        print("Q网络隐藏层：", hd)
        wandb.init(
            project="LAG-SingleControl-hidden_dim_train4eval",
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
        seed = 42
        print("-"*80)
        print("随机种子：", seed)
        returns = main(
                    learning_rate, num_episodes, hd, gamma, epsilon,
                    tau, buffer_size, minimal_size, batch_size, device, improvements, 
                    seed, model_save_path, save_episode_place
                    )
        returns = returns[500:]  #统计在后半段episode时的episode return，来计算基本训练完成时的稳定性。
    
        variance = np.var(returns)
        std = np.std(returns)
        mean = np.mean(returns)
        wandb.log({
            "variance": variance,
            "std": std,
            "mean": mean
        })

        wandb.finish()