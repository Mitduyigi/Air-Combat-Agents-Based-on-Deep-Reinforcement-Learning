import torch
import random
import numpy as np
import logging
from pathlib import Path
import argparse

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.dqn import DQN  
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from LAG.envs.JSBSim.envs.singlecontrol_env import SingleControlEnv


@torch.no_grad()
def render_model(seed, model_name):
    env = SingleControlEnv("1/heading")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.seed(seed)

    if torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")

    # run dir
    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/render_results")

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    state_dim = env.observation_space.shape[0]
    action_dims = list(env.action_space.nvec)
    hidden_dim = int(model_name.split("_")[4]) if "hidden_dim" in model_name else 64
    improvements = model_name.split("_")[3] if "improvements" in model_name else []
    agent = DQN(state_dim, hidden_dim, action_dims, epsilon=0.0, improvements=improvements)
    agent.q_net.load_state_dict(torch.load(model_name))
    agent.q_net.eval()  

    simple_model_name = model_name.split("/")[4].split(".")[0]
    logging.info("\nStart render ...")
    render_episode_rewards = 0
    render_obs = env.reset()
    env.render(mode='txt', filepath=f'{run_dir}/{simple_model_name}.acmi')
    while True:
        render_actions = [agent.take_action(render_obs)]
        render_obs, render_rewards, render_dones, render_infos = env.step(render_actions)
        render_episode_rewards += render_rewards
        env.render(mode='txt', filepath=f'{run_dir}/{simple_model_name}.acmi')
        if render_dones.all():
            break
    render_infos = {}
    render_infos['render_episode_reward'] = render_episode_rewards
    logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))
    env.close()
    logging.info(f"acmi files save place:{run_dir}/{simple_model_name}.acmi")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render model script")
    parser.add_argument("--seed", type=int, required=True, help="Seed for rendering")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the model file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    render_model(args.seed, args.model_name)