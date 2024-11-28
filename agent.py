import torch
import torch.optim as optim
import torch.nn as nn
import random
from model.pointer_network import PointerNetwork

from config import *

class Agent:
    def __init__(self, age_group='adult', gender='male', health_status='good'):
        self.age_group = age_group
        self.gender = gender
        self.health_status = health_status
        self.set_speed_and_explore_ratio()
        
    def set_speed_and_explore_ratio(self):
        # 기본 탐험률과 속도 확률 설정
        if self.age_group == 'young':  # 20대
            self.explore_ratio = 0.3
            self.speed_probabilities = [0.2, 0.5, 0.3]  # [느림, 보통, 빠름]
        elif self.age_group == 'middle':  # 40-50대
            self.explore_ratio = 0.2
            self.speed_probabilities = [0.3, 0.4, 0.3]
        else:  # 'old' - 60대 이상
            self.explore_ratio = 0.1
            self.speed_probabilities = [0.5, 0.3, 0.2]
        
        # 건강상태에 따른 조정
        if self.health_status == 'bad':
            self.explore_ratio *= 0.7
            # 느린 속도의 확률 증가
            self.speed_probabilities = [0.6, 0.3, 0.1]
        elif self.health_status == 'good':
            self.explore_ratio *= 1.2
            # 빠른 속도의 확률 증가
            self.speed_probabilities = [0.2, 0.4, 0.4]
            
        # 성별에 따른 조정
        if self.gender == 'female':
            self.explore_ratio *= 0.9
            
class PointerDQN:
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=8, learning_rate=0.001, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PointerNetwork(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_network = PointerNetwork(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
    def get_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, 7)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values, _ = self.network(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values, _ = self.network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_states)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * CONFIG['gamma'] * next_q_values
        
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())
    
    def save(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])