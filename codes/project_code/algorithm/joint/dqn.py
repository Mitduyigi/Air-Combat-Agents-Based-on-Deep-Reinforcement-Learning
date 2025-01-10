import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import collections
from copy import deepcopy

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done):  
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  
        return len(self.buffer)
    
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, episilon=1e-6):
        self.buffer = collections.deque(maxlen=capacity) 
        self.priorities = collections.deque(maxlen=capacity) 
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.episilon = episilon
        self.empty = True

    def add(self, state, action, reward, next_state, done):  
        if self.empty:
            priority = 1.0
            self.empty = False
        else:
            priority = max(self.priorities) 
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)

    def sample(self, batch_size): 
        priorities = self.priorities

        probabilities = np.array(priorities) / sum(priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        transitions = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, np.array(reward), np.array(next_state), np.array(done), indices, weights
    
    def update_priorities(self, indices, td_errors):
        reshaped_td_errors = [sum(td_error_act[i][0].item() for td_error_act in td_errors) for i in range(64)] #直接把四个维度的td error直接加起来是否合理？
        for idx, td_error in zip(indices, reshaped_td_errors):
            self.priorities[int(idx)] = (np.abs(td_error) + self.episilon) ** self.alpha

    def size(self):  
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
class VAnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.advantage_stream = nn.Linear(hidden_dim, action_dim)
        self.value_stream = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        advantages = self.advantage_stream(x)
        value = self.value_stream(x)
        Q = value + advantages - advantages.mean(dim=-1, keepdim=True)
        return Q
    
class DQN:
    def __init__(self, state_dim, hidden_dim, action_dims, 
                 learning_rate, gamma, epsilon, target_update, capacity, device, improvements=[]):
        self.action_dims = action_dims
        self.process_action_spaces(action_dims)

        if "DuelingDQN" in improvements:
            self.q_net = VAnet(state_dim, hidden_dim, self.joint_action_dim).to(device)
            self.target_q_net = VAnet(state_dim, hidden_dim, self.joint_action_dim).to(device)
        else:
            self.q_net = Qnet(state_dim, hidden_dim, self.joint_action_dim).to(device) 
            self.target_q_net = Qnet(state_dim, hidden_dim, self.joint_action_dim).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.target_update = target_update  
        if "PrioritizedDQN" in improvements:
            self.replay_buffer = PrioritizedReplayBuffer(capacity)
        else:
            self.replay_buffer = ReplayBuffer(capacity)
        self.count = 0  
        self.device = device
        self.improvements = improvements
        print("using improvements:", [improvement for improvement in improvements])
        print("-"*160)

    def process_action_spaces(self, action_dims):
            self.joint_action_dim = 1
            for space in action_dims:
                self.joint_action_dim *= space

    def joint_to_indiv(self, orig_action):
        """Convert joint action to individual actions."""
        action = deepcopy(orig_action)
        actions = []
        for dim in self.action_dims:
            actions.append(action % dim)
            action = int(torch.div(action, dim, rounding_mode="floor"))
        return actions
    
    def indiv_to_joint(self, orig_actions):   
        """Convert individual actions to joint action."""
        actions = deepcopy(orig_actions)
        action = torch.zeros_like(torch.tensor(actions[0]))
        accum_dim = 1
        for i, dim in enumerate(self.action_dims):
            action += accum_dim * actions[i]
            accum_dim *= dim
        return int(action)
    
    def take_action(self, state):  
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.joint_action_dim)
        else:
            state = torch.from_numpy(state).to(torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        if "PrioritizedDQN" in self.improvements:
            indices = torch.tensor(transition_dict['indices'], dtype=torch.float).to(self.device)
            weights = torch.tensor(transition_dict['weights'], dtype=torch.float).to(self.device)
        
        print(actions.shape)
        print(self.q_net(states).shape)
        q_values = self.q_net(states).gather(1, actions)

        if "DoubleDQN" in self.improvements:
            max_action = []
            for i in range(actions.shape[1]):
                # .max(1)：在维度 1 (动作维度) 上选取 Q 值最大的动作。返回值为一个元组 (max_values, indices)
                max_action.append(self.q_net(next_states)[i].max(2)[1].view(-1, 1)) #(64,1)   
            max_next_q_values = []
            for i in range(actions.shape[1]):
                max_next_q_values.append(self.target_q_net(next_states)[i].squeeze(1).gather(1, max_action[i])) 
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        if "PrioritizedDQN" in self.improvements:
            td_errors = []
            for i in range(actions.shape[1]):
                td_errors.append(q_targets[i] - q_values[i])
            self.replay_buffer.update_priorities(np.array(indices), td_errors)
            loss = []
            for i in range(actions.shape[1]):
                loss.append((0.5 * weights * td_errors[i] ** 2).mean())
        else:
            loss = torch.mean(F.mse_loss(q_values, q_targets))
        
        self.optimizer.zero_grad()
        loss.backward()   
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  
        self.count += 1