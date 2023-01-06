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

# Ornstein Uhlenbeck Action Noise
class OUActionNoise(object): # this class is a reference to the content of Openai baseline.
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()
    
    def __call__(self):
        x = self.x_prev + self.theta*(self.mu - self.x_prev)*self.dt + self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

# Deep Deterministic Policy Gradient Network
class ActorNNs(nn.Module):
    def __init__(self, alphaA, input_dims, fc1_dims, fc2_dims, n_actions): 
        super(ActorNNs, self).__init__()
        self.alphaA = alphaA
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.actor = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.alphaA)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = T.tanh(self.actor(x))
        return action
        
class CriticNNs(nn.Module):
    def __init__(self, alphaC, input_dims, fc1_dims, fc2_dims, n_actions):
        super(CriticNNs, self).__init__()
        self.alphaC = alphaC
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims + self.n_actions, self.fc2_dims)
        self.critic = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.alphaC)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(T.cat([x, action], -1)))
        Qvalue = self.critic(x)
        return Qvalue

# Agent
class Agent(object):
    def __init__(self, gamma, alphaA, alphaC, tau,
                 input_dims, n_actions, fc1_dims, fc2_dims, 
                 memory_size, batch_size):
        self.gamma = gamma
        self.alphaA = alphaA
        self.alphaC = alphaC
        self.tau = tau
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        
        self.actor = ActorNNs(alphaA, input_dims, fc1_dims, fc2_dims, n_actions)
        self.critic = CriticNNs(alphaC, input_dims, fc1_dims, fc2_dims, n_actions)
        self.targetActor = ActorNNs(alphaA, input_dims, fc1_dims, fc2_dims, n_actions)
        self.targetCritic = CriticNNs(alphaC, input_dims, fc1_dims, fc2_dims, n_actions)
    
    def select_action(self, observation):
        state = T.Tensor(observation).to(self.actor.device)
        action = self.actor(state) + T.tensor(self.noise()).to(self.actor.device)
        return action.cpu().detach().numpy()
    
    def push(self, state, action, next_state, reward, done):
        self.memory.store_transition(state, action, next_state, reward, done)
    
    def soft_target_update(self):
        a_params = self.actor.named_parameters()
        tA_params = self.targetActor.named_parameters()
        c_params = self.critic.named_parameters()
        tC_params = self.targetCritic.named_parameters()
        a_state_dict = dict(a_params)
        tA_state_dict = dict(tA_params)
        c_state_dict = dict(c_params)
        tC_state_dict = dict(tC_params)
        
        with T.no_grad():
            for name in a_state_dict:
                a_state_dict[name] = (1.0 - self.tau)*tA_state_dict[name].clone() + self.tau*a_state_dict[name].clone()
        self.targetActor.load_state_dict(a_state_dict) 
        
        with T.no_grad():
            for name in c_state_dict:
                c_state_dict[name] = (1.0 - self.tau)*tC_state_dict[name].clone() + self.tau*c_state_dict[name].clone()
        self.targetCritic.load_state_dict(c_state_dict)
        
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
        
        criticValue = self.critic(state, action)
        targetAction = self.targetActor(state_)
        criticValue_ = self.targetCritic(state_, targetAction)
        criticValue_[done] = 0.0
        
        y = reward + self.gamma*criticValue_
        self.critic.optimizer.zero_grad()
        criticLoss = F.mse_loss(y, criticValue)
        criticLoss.backward()
        self.critic.optimizer.step()
        
        self.actor.optimizer.zero_grad()
        action = self.actor(state)
        actorLoss = -self.critic(state, action)
        actorLoss = T.mean(actorLoss)
        actorLoss.backward()
        self.actor.optimizer.step()
        
        self.soft_target_update()
        
        Loss = criticLoss + actorLoss
        
        return Loss

# Main 
if __name__ == '__main__':
    
    env = PandaEnv1()
    env.seed(0)
    
    wandb.init(project="Paper1", name="DDPG")
    
    agent = Agent(gamma=..., alphaA=..., alphaC=..., tau=...,
                 input_dims=env.observation_space.shape[0], n_actions=env.action_space.shape[0], 
                 fc1_dims=..., fc2_dims=..., 
                 memory_size=..., batch_size=...)
    
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
        agent.critic.load_state_dict(checkpoint['state_dict2'])
        agent.critic.optimizer.load_state_dict(checkpoint['optimizer2'])
        agent.targetActor.load_state_dict(checkpoint['state_dict3'])
        agent.targetActor.optimizer.load_state_dict(checkpoint['optimizer3'])
        agent.targetCritic.load_state_dict(checkpoint['state_dict4'])
        agent.targetCritic.optimizer.load_state_dict(checkpoint['optimizer4'])
    
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
        accuracy_rate = ep_Accuracy/100
        safe_rate = (ep_safe_rate)/100
        wandb.log({'reward': score, 'accuracy': accuracy_rate, 'Loss': ep_Loss, 'safe_rate': safe_rate})
        
        print('episode', i+1, 'reward %.1f' % score, 'avg_reward %.1f' % avg_score, 
              'loss_avg %.1f' % loss_avg,'accuracy %.2f' % accuracy_rate, 'safe_rate %.2f' % safe_rate)
        
        if score == np.max(score_history):
            checkpoint = {'epoch': n_episodes+1,
                      'state_dict1': agent.actor.state_dict(), 
                      'optimizer1': agent.actor.optimizer.state_dict(),
                      'state_dict2': agent.critic.state_dict(), 
                      'optimizer2': agent.critic.optimizer.state_dict(),
                      'state_dict3': agent.targetActor.state_dict(), 
                      'optimizer3': agent.targetActor.optimizer.state_dict(),
                      'state_dict4': agent.targetCritic.state_dict(), 
                      'optimizer4': agent.targetCritic.optimizer.state_dict()}
            save_checkpoint(checkpoint)
            
    env.render()
    env.close()  