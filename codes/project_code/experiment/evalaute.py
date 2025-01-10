import os
import re
import torch
import numpy as np
import random
from tqdm import tqdm
import wandb

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.dqn import DQN  
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from LAG.envs.JSBSim.envs.singlecontrol_env import SingleControlEnv


def parse_model_file(file_name):
    patterns = {
        "learning_rate": r"learning_rate_([\d.eE-]+)",  # 匹配 learning_rate 的值
        "hidden_dim": r"hidden_dim_(\d+)",              # 匹配 hidden_dim 的值
        "improvements": r"improvements_(\[[^\]]*\])",    # 匹配 improvements 的值
    }
    parameters = {}
    for param, pattern in patterns.items():
        match = re.search(pattern, file_name)
        if match:
            parameters[param] = match.group(1)
    return parameters


def load_models(models_folder):
    if not os.path.exists(models_folder):
        raise FileNotFoundError(f"Folder '{models_folder}' does not exist.")
    file_names = []
    parameters = []
    for file_name in os.listdir(models_folder):
        if file_name.endswith(".pt"):
            parameter = parse_model_file(file_name)
            file_names.append(file_name)
            parameters.append(parameter)
    return file_names, parameters

def preprocess_models(file_names, parameters):
    learning_rate_models = {"0.001": [], "0.0005": [], "0.0001": [], "5e-05": []}
    hidden_dim_models = {"128": [], "64": [], "32": []}
    improvements_models = {"[]": [], "['DoubleDQN']": [], "['PrioritizedDQN']": [], "['DuelingDQN']": [], 
                           "['DoubleDQN', 'PrioritizedDQN', 'DuelingDQN']": [],
                           "['DoubleDQN', 'PrioritizedDQN']": [], 
                           "['DoubleDQN', 'DuelingDQN']": [], 
                           "['PrioritizedDQN', 'DuelingDQN']": []} 

    for file_name, param in zip(file_names, parameters):
        learning_rate = param.get("learning_rate", None)
        hidden_dim = param.get("hidden_dim", None)
        improvements = param.get("improvements", None)  
        if learning_rate in learning_rate_models:
            learning_rate_models[learning_rate].append(file_name)
        if hidden_dim in hidden_dim_models:
            hidden_dim_models[hidden_dim].append(file_name)
        if improvements in improvements_models:
            improvements_models[improvements].append(file_name)
    return learning_rate_models, hidden_dim_models, improvements_models


@torch.no_grad()
def evaluate_model(model_path, num_episodes, seed, hidden_dim=64, learning_rate=1e-4, improvements=[]):
    env = SingleControlEnv("1/heading")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dims = list(env.action_space.nvec)

    agent = DQN(state_dim, hidden_dim, action_dims, learning_rate, epsilon=0.0, improvements=improvements)
    agent.q_net.load_state_dict(torch.load(model_path))
    agent.q_net.eval()  

    return_list = []
    with tqdm(total=num_episodes, desc="Evaluating") as pbar:
        for episode in range(num_episodes):
            episode_return = 0
            state = env.reset()  
            done = False
            while not done:
                actions = [agent.take_action(state)]
                next_state, reward, done, _ = env.step(actions)
                episode_return += reward
                state = next_state
            wandb.log({'episode': episode, 'reward': episode_return})
            return_list.append(episode_return)
            if (episode + 1) % 10 == 0:
                pbar.set_postfix({'Episode': episode + 1, 'Reward': np.mean(return_list[-5:])})
            pbar.update(1)

    variance = np.var(return_list)
    std = np.std(return_list)
    mean = np.mean(return_list)
    wandb.log({
        "variance": variance, 
        "std": std, 
        "mean": mean
    })
    env.close()
    return mean, std, variance


if __name__ == '__main__':
    num_episodes = 500  
    gamma = 0.99  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    seed = 42
    models_folder = "train_for_evaluate/models"
    file_names, parameters = load_models(models_folder)
    learning_rate_models, hidden_dim_models, improvements_models = preprocess_models(file_names, parameters)


    for learning_rate, models in learning_rate_models.items():
        wandb.init(project="LAG-SingleControl-learning_rate_evaluate", 
                   config={"learning_rate": learning_rate}, mode="offline")
        print(f"Evaluating models with learning_rate = {learning_rate}")
        learning_rate = float(learning_rate)
        mean_list = []
        std_list = []
        variance_list = []
        for model in models:
            model_path = os.path.join(models_folder, model)
            mean, std, variance = evaluate_model(model_path, num_episodes, seed, learning_rate=learning_rate)
            mean_list.append(mean)
            std_list.append(std)
            variance_list.append(variance)
        wandb.finish()
        wandb.init(project="LAG-SingleControl-learning_rate_evaluate", 
                   config={"learning_rate": learning_rate}, mode="offline")
        wandb.log({
            "variance": np.mean(mean_list), 
            "std": np.mean(std_list), 
            "mean": np.mean(variance_list)
        })
        wandb.finish()


    for hidden_dim, models in hidden_dim_models.items():
        wandb.init(project="LAG-SingleControl-hidden_dim_evaluate", 
                   config={"hidden_dim": hidden_dim}, mode="offline")
        print(f"Evaluating models with hidden_dim = {hidden_dim}")
        hidden_dim = int(hidden_dim)
        mean_list = []
        std_list = []
        variance_list = []
        for model in models:
            model_path = os.path.join(models_folder, model)
            mean, std, variance = evaluate_model(model_path, num_episodes, seed, hidden_dim=hidden_dim)
            mean_list.append(mean)
            std_list.append(std)
            variance_list.append(variance)
        wandb.finish()
        wandb.init(project="LAG-SingleControl-hidden_dim_evaluate", 
                   config={"hidden_dim": hidden_dim}, mode="offline")
        wandb.log({
            "variance": np.mean(mean_list), 
            "std": np.mean(std_list), 
            "mean": np.mean(variance_list)
        })
        wandb.finish()


    for improvements, models in improvements_models.items():
        wandb.init(project="LAG-SingleControl-improvements_evaluate", 
                   config={"improvements": improvements}, mode="offline")
        print(f"Evaluating models with improvements = {improvements}")
        mean_list = []
        std_list = []
        variance_list = []
        for model in models:
            model_path = os.path.join(models_folder, model)
            mean, std, variance = evaluate_model(model_path, num_episodes, seed, improvements=improvements)
            mean_list.append(mean)
            std_list.append(std)
            variance_list.append(variance)
        wandb.finish()
        wandb.init(project="LAG-SingleControl-improvements_evaluate", 
                   config={"improvements": improvements}, mode="offline")
        wandb.log({
            "variance": np.mean(mean_list), 
            "std": np.mean(std_list), 
            "mean": np.mean(variance_list)
        })
        wandb.finish()