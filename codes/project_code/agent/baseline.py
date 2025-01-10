import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from typing import Literal
import argparse
import wandb

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from LAG.envs.JSBSim.core.catalog import Catalog as c
from LAG.envs.JSBSim.utils.utils import in_range_rad
from LAG.envs.JSBSim.envs import SingleCombatEnv
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from algorithm.dqn import DQN


class BaselineAgent(DQN):
    def __init__(self, agent_id, state_dim, hidden_dim, action_dims, improvements, model_name) -> None:
        super().__init__(state_dim, hidden_dim, action_dims, epsilon=0.0, improvements=improvements)
        self.model_path = os.path.dirname(os.path.realpath(__file__)) + "/" + model_name
        self.q_net.load_state_dict(torch.load(self.model_path))
        self.q_net.eval()
        self.agent_id = agent_id
        self.state_var = [
            c.delta_altitude,                   #  0. delta_h   (unit: m)
            c.delta_heading,                    #  1. delta_heading  (unit: °)
            c.delta_velocities_u,               #  2. delta_v   (unit: m/s)
            c.attitude_roll_rad,                #  3. roll      (unit: rad)
            c.attitude_pitch_rad,               #  4. pitch     (unit: rad)
            c.velocities_u_mps,                 #  5. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 #  6. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 #  7. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                #  8. vc        (unit: m/s)
            c.position_h_sl_m                   #  9. altitude  (unit: m)
        ]
        self.reset()

    def reset(self):
        pass

    @abstractmethod
    def set_delta_value(self, observation):
        raise NotImplementedError

    def get_observation(self, env, task, delta_value):
        uid = list(env.agents.keys())[self.agent_id]
        obs = env.agents[uid].get_property_values(self.state_var)
        norm_obs = np.zeros(12)
        norm_obs[0] = delta_value[0] / 1000          #  0. ego delta altitude  (unit: 1km)
        norm_obs[1] = in_range_rad(delta_value[1])   #  1. ego delta heading   (unit rad)
        norm_obs[2] = delta_value[2] / 340           #  2. ego delta velocities_u  (unit: mh)
        norm_obs[3] = obs[9] / 5000                  #  3. ego_altitude (unit: km)
        norm_obs[4] = np.sin(obs[3])                 #  4. ego_roll_sin
        norm_obs[5] = np.cos(obs[3])                 #  5. ego_roll_cos
        norm_obs[6] = np.sin(obs[4])                 #  6. ego_pitch_sin
        norm_obs[7] = np.cos(obs[4])                 #  7. ego_pitch_cos
        norm_obs[8] = obs[5] / 340                   #  8. ego_v_x   (unit: mh)
        norm_obs[9] = obs[6] / 340                   #  9. ego_v_y    (unit: mh)
        norm_obs[10] = obs[7] / 340                  #  10. ego_v_z    (unit: mh)
        norm_obs[11] = obs[8] / 340                  #  11. ego_vc        (unit: mh)
        norm_obs = np.expand_dims(norm_obs, axis=0)  # dim: (1,12)
        return norm_obs

    def get_action(self, env, task):
        delta_value = self.set_delta_value(env, task)
        observation = self.get_observation(env, task, delta_value)
        observation = torch.from_numpy(observation).to(torch.float)
        actions = [action_score.argmax().item() for action_score in self.q_net(observation)]
        return actions


class PursueAgent(BaselineAgent):
    def __init__(self, agent_id, state_dim, hidden_dim, action_dims, improvements, model_name) -> None:
        super().__init__(agent_id, state_dim, hidden_dim, action_dims, improvements=improvements, model_name=model_name)

    def set_delta_value(self, env, task):
        # NOTE: only adapt for 1v1
        ego_uid, enm_uid = list(env.agents.keys())[self.agent_id], list(env.agents.keys())[(self.agent_id+1)%2] 
        ego_x, ego_y, ego_z = env.agents[ego_uid].get_position()
        ego_vx, ego_vy, ego_vz = env.agents[ego_uid].get_velocity()
        enm_x, enm_y, enm_z = env.agents[enm_uid].get_position()
        # delta altitude
        delta_altitude = enm_z - ego_z
        # delta heading
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        delta_heading = ego_AO * side_flag
        # delta velocity
        delta_velocity = env.agents[enm_uid].get_property_value(c.velocities_u_mps) - \
                         env.agents[ego_uid].get_property_value(c.velocities_u_mps)
        return np.array([delta_altitude, delta_heading, delta_velocity])


class ManeuverAgent(BaselineAgent):
    def __init__(self, agent_id, state_dim, hidden_dim, action_dims, improvements, maneuver: Literal['l', 'r', 'n'], model_name) -> None:
        super().__init__(agent_id, state_dim, hidden_dim, action_dims, improvements=improvements, model_name=model_name)
        self.turn_interval = 30
        self.dodge_missile = False # if set true, start turn when missile is detected
        if maneuver == 'l':
            self.target_heading_list = [0]
        elif maneuver == 'r':
            self.target_heading_list = [np.pi/2, np.pi/2, np.pi/2, np.pi/2]
        elif maneuver == 'n':
            self.target_heading_list = [np.pi, np.pi, np.pi, np.pi]
        elif maneuver == 'triangle':
            self.target_heading_list = [np.pi/3, np.pi, -np.pi/3]*2
        elif maneuver == 'square':
            self.target_heading_list = [np.pi/4, np.pi*3/4, -np.pi*3/4, -np.pi/4]*2
        elif maneuver == 'pentagram':
            self.target_heading_list = [(i * (2 * np.pi / 5)) % (2 * np.pi) for i in [0, 2, 4, 1, 3]]*2
        self.target_altitude_list = [6000] * 6
        self.target_velocity_list = [243]  * 6

    def reset(self):
        self.step = 0
        self.init_heading = None

    def set_delta_value(self, env, task):
        step_list = np.arange(1, len(self.target_heading_list)+1) * self.turn_interval / env.time_interval
        uid = list(env.agents.keys())[self.agent_id]
        cur_heading = env.agents[uid].get_property_value(c.attitude_heading_true_rad)
        if self.init_heading is None:
            self.init_heading = cur_heading
        if not self.dodge_missile or task._check_missile_warning(env, self.agent_id) is not None:
            for i, interval in enumerate(step_list):
                if self.step <= interval:
                    break
            delta_heading = self.init_heading + self.target_heading_list[i] - cur_heading
            delta_altitude = self.target_altitude_list[i] - env.agents[uid].get_property_value(c.position_h_sl_m)
            delta_velocity = self.target_velocity_list[i] - env.agents[uid].get_property_value(c.velocities_u_mps)
            self.step += 1
        else:
            delta_heading = self.init_heading  - cur_heading
            delta_altitude = 6000 - env.agents[uid].get_property_value(c.position_h_sl_m)
            delta_velocity = 243 - env.agents[uid].get_property_value(c.velocities_u_mps)

        return np.array([delta_altitude, delta_heading, delta_velocity])


def test_maneuver(seed, model_name, maneuver):
    only_model_name = model_name.split("/")[1]

    env = SingleCombatEnv(config_name='1v1/NoWeapon/test/opposite')
    env.seed(seed)
    obs = env.reset()
    env.render(filepath=f'output/{maneuver}_{only_model_name}.acmi')
    
    '''state_dim = env.observation_space.shape[0]
    action_dims = list(env.action_space.nvec)'''
    state_dim = 12
    action_dims = [41, 41, 41, 30]
    hidden_dim = int(model_name.split("_")[4]) if "hidden_dim" in model_name else 64
    improvements = model_name.split("_")[3] if "improvements" in model_name else []

    agent0 = ManeuverAgent(0, state_dim, hidden_dim, action_dims, improvements, maneuver, model_name=model_name)
    agent1 = PursueAgent(1, state_dim, hidden_dim, action_dims, improvements, model_name=model_name)
    reward_list = []
    while True:
        action0 = agent0.get_action(env, env.task)
        action1 = agent1.get_action(env, env.task)
        actions = [action0, action1]
        obs, reward, done, info = env.step(actions)
        env.render(filepath=f'output/{maneuver}_{only_model_name}.acmi')
        if np.array(done).all():
            print(info)
            break
        wandb.log({
            'agent0 reward': reward[0][0],
            'agent1 reward': reward[1][0]
        })


if __name__ == '__main__':
    if not os.path.exists("output/"):
        os.makedirs("output/")

    '''parser = argparse.ArgumentParser(description="Render model script")
    parser.add_argument("--seed", type=int, required=True, help="Seed for rendering")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the model file")
    parser.add_argument("--maneuver", type=str, required=True, help="Maneuver type")
    args = parser.parse_args()

    test_maneuver(args.seed, args.model_name, args.maneuver)'''
    wandb.init(
        project="LAG-maneuver_pursue-baseline",
        mode="offline",
    )
    model_name = 'models/best_1000.0_20241229-013305.pt'
    test_maneuver(42, model_name, 'triangle')
    wandb.finish()