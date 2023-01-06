import pybullet_envs
import gym
from env import PandaEnv1
import math
import random 
import numpy as np    
from collections import namedtuple, deque
from tqdm import tqdm
import wandb
import time
import os

import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print('Using device:', device)

# Additional Info when using cuda
if device.type == 'cuda':
    print(T.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(T.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(T.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
T.set_default_dtype(T.float32) 

#  Replay Buffer
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer(object): 
    def __init__(self, memory_capacity):
        self.memory = deque([],maxlen=memory_capacity)
        
    def store_transition(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)

# Soft Actor Critic Network
class ActorNNs(nn.Module):
    def __init__(self, alphaA, input_dims, fc1_dims, fc2_dims, n_actions): 
        super(ActorNNs,self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions 
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mean = nn.Linear(self.fc2_dims, self.n_actions)
        self.std = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alphaA)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = self.std(x)
        std = T.clamp(std, min=..., max=...)
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        pi_distribution = Normal(mean, std) 
        
        # leverage reparameterization trick here, in order to reduce variance
        a = pi_distribution.rsample()
        action = T.tanh(a).to(self.device)
        
        log_prob = pi_distribution.log_prob(a)
        log_prob -= T.log(1-action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True) 
        
        return action, log_prob

class CriticNNs(nn.Module):
    def __init__(self, alphaC, input_dims, fc1_dims, fc2_dims, n_actions): 
        super(CriticNNs, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(self.input_dims + self.n_actions, self.fc1_dims) 
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.Qvalue = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alphaC)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], 1))) 
        x = F.relu(self.fc2(x))
        Qvalue = self.Qvalue(x)
        return Qvalue
        
class ValueNNs(nn.Module):
    def __init__(self, alphaV, input_dims, fc1_dims, fc2_dims):
        super(ValueNNs, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims) 
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.value = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alphaV)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value

# Agent
class Agent(object):
    def __init__(self, gamma, alphaA, alphaC, alphaV, tau,
                 input_dims, n_actions, fc1_dims, fc2_dims, 
                 memory_size, batch_size, reward_scale):
        self.gamma = gamma
        self.alphaA = alphaA 
        self.alphaC = alphaC 
        self.alphaV = alphaV 
        self.tau = tau
        self.input_dims = input_dims 
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims 
        self.fc2_dims = fc2_dims 
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.reward_scale = reward_scale 
        
        self.actor = ActorNNs(alphaA, input_dims, fc1_dims, fc2_dims, n_actions) 
        self.critic1 = CriticNNs(alphaC, input_dims, fc1_dims, fc2_dims, n_actions)
        self.critic2 = CriticNNs(alphaC, input_dims, fc1_dims, fc2_dims, n_actions)
        self.value = ValueNNs(alphaV, input_dims, fc1_dims, fc2_dims)
        self.targetValue = ValueNNs(alphaV, input_dims, fc1_dims, fc2_dims)
        
    def select_action(self, observation):
        state = T.Tensor(observation).to(self.actor.device)
        action, _ = self.actor.sample(state)
        return action.cpu().detach().numpy()  
        
    def push(self, state, action, next_state, reward, done):
        self.memory.store_transition(state, action, next_state, reward, done)
        
    def soft_update(self):
        V_params = self.value.named_parameters()
        tV_params = self.targetValue.named_parameters()
        v_state_dict = dict(V_params)
        tV_state_dict = dict(tV_params)
        
        with T.no_grad():
            for name in v_state_dict:
                v_state_dict[name] = (1.0 - self.tau)*tV_state_dict[name].clone() + self.tau*v_state_dict[name].clone()
        self.targetValue.load_state_dict(v_state_dict)
        
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size) 
        batch = Transition(*zip(*transitions)) 
        
        state = T.tensor(batch.state, dtype=T.float).to(self.actor.device)
        action = T.tensor(batch.action, dtype=T.float).to(self.actor.device)
        state_ = T.tensor(batch.next_state, dtype=T.float).to(self.actor.device)
        reward = T.tensor(batch.reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(batch.done).to(self.actor.device)
        reward = T.reshape(reward, (self.batch_size,1)) 
        done = T.reshape(done, (self.batch_size,1)) 
        
        value = self.value(state)
        value_ = self.targetValue(state_)
        value_[done] = 0.0
        
        pi, log_pi = self.actor.sample(state)
        new_q1pi = self.critic1.forward(state, pi)
        new_q2pi = self.critic2.forward(state, pi)
        min_qpi = T.min(new_q1pi, new_q2pi)
        
        self.value.optimizer.zero_grad()
        valueTarget = min_qpi - log_pi
        valueLoss = 0.5 * F.mse_loss(value, valueTarget)
        valueLoss.backward(retain_graph=True)
        self.value.optimizer.step()

        actorLoss = T.mean(log_pi - min_qpi)
        self.actor.optimizer.zero_grad()
        actorLoss.backward(retain_graph=True)
        self.actor.optimizer.step()
        
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        q_hat = self.reward_scale*reward + self.gamma*value_
        old_q1pi = self.critic1.forward(state, action)
        old_q2pi = self.critic2.forward(state, action)
        critic1Loss = 0.5 * F.mse_loss(old_q1pi, q_hat)
        critic2Loss = 0.5 * F.mse_loss(old_q2pi, q_hat)
        
        criticLoss = critic1Loss + critic2Loss
        criticLoss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()
        self.soft_update()
        
        Loss = actorLoss + criticLoss
        
        return Loss

# Main 
if __name__ == '__main__':
     
    env = PandaEnv1()
    env.seed(0)
    
    wandb.init(project="Paper1", name="SoftActorCritic")
    
    agent = Agent(gamma=..., alphaA=..., alphaC=..., alphaV=..., tau=..., 
                  input_dims=env.observation_space.shape[0], n_actions=env.action_space.shape[0],
                  fc1_dims=..., fc2_dims=..., 
                  memory_size=..., batch_size=..., reward_scale=...)
    
    # Training till fill up Replay Buffer to batch size
    step = 0
    while step < agent.batch_size:
        obs = env.reset()
        done = False
        while not done:
            step += 1
            action = env.action_space.sample()
            obs_, reward, done, info, accuracy, safe_rate = env.step(action)
            agent.push(obs, action, obs_, reward, done)
            obs = obs_
    
    def save_checkpoint(state, filename = "best_model.pt"):
        print("=> Saving checkpoint")
        T.save(state, filename)
    
    def load_checkpoint(checkpoint):
        print("=> Loading checkpoint")
        agent.actor.load_state_dict(checkpoint['state_dict1'])
        agent.actor.optimizer.load_state_dict(checkpoint['optimizer1'])
        agent.critic1.load_state_dict(checkpoint['state_dict2'])
        agent.critic1.optimizer.load_state_dict(checkpoint['optimizer2'])
        agent.critic2.load_state_dict(checkpoint['state_dict3'])
        agent.critic2.optimizer.load_state_dict(checkpoint['optimizer3'])
        agent.value.load_state_dict(checkpoint['state_dict4'])
        agent.value.optimizer.load_state_dict(checkpoint['optimizer4'])
        agent.targetValue.load_state_dict(checkpoint['state_dict5'])
        agent.targetValue.optimizer.load_state_dict(checkpoint['optimizer5'])
    
    n_episodes = 10000
    score_history = []
    loss_history = []
    load_model = False
    
    if load_model:
        localPath = r"C:\Users\..." 
        load_checkpoint(T.load(os.path.join(localPath,"best_model.pt")))
    
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        score = 0
        ep_Loss = 0
        ep_Accuracy = 0
        ep_safe_rate = 0
        while not done:
            action = agent.select_action(obs)
            obs_, reward, done, info, accuracy, safe_rate = env.step(action)
            score += reward
            agent.push(obs, action, obs_, reward, done)
            Loss = agent.learn()
            ep_Loss += Loss
            ep_Accuracy += accuracy
            ep_safe_rate += safe_rate
            obs = obs_
            loss_history.append(Loss.item())
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        loss_avg = np.mean(loss_history)
        accuracy_rate = (ep_Accuracy)/100
        safe_rate = (ep_safe_rate)/100
        wandb.log({'reward': score, 'accuracy': accuracy_rate, 'Loss': ep_Loss, 'safe_rate': safe_rate})
        
        print('episode', i+1, 'reward %.1f' % score, 'avg_reward %.1f' % avg_score, 
              'loss_avg %.1f' % loss_avg,'accuracy %.2f' % accuracy_rate, 'safe_rate %.2f' % safe_rate)
        
        if score == np.max(score_history):
            checkpoint = {'epoch': n_episodes+1,
                      'state_dict1': agent.actor.state_dict(), 
                      'optimizer1': agent.actor.optimizer.state_dict(),
                      'state_dict2': agent.critic1.state_dict(), 
                      'optimizer2': agent.critic1.optimizer.state_dict(),
                      'state_dict3': agent.critic2.state_dict(), 
                      'optimizer3': agent.critic2.optimizer.state_dict(),
                      'state_dict4': agent.value.state_dict(), 
                      'optimizer4': agent.value.optimizer.state_dict(),
                      'state_dict5': agent.targetValue.state_dict(), 
                      'optimizer5': agent.targetValue.optimizer.state_dict()}
            save_checkpoint(checkpoint)
            
    env.render()
    env.close()             