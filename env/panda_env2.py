import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import pybullet_data

import os
import math
import numpy as np
import random
import time
import operator
from scipy.spatial import distance

MAX_EPISODE_LEN = 100

class PandaEnv2(gym.Env):
    metadata = {'render.modes': ['human'],
                'video.frames_per_second': 60}
    
    def __init__(self, is_render=True, is_good_view=False):
        self.step_counter = 0
        self.is_render = is_render
        
        # Slow motion for better observation 
        self.is_good_view = is_good_view
        
        # Render or not
        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        
        # Setup Camera view position and orientation 
        p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=135, cameraPitch=-30, 
                                     cameraTargetPosition=[0.25,-0.2,-0.3])
        
        # Setup action and observation space 
        n_action = 3 # move along x, y, z axis
        n_observation = 30 # End-effector & Target & Obstacles's position and velocity 
        self.action_space = spaces.Box(low=np.array([-1]*n_action, dtype=np.float32), 
                                       high=np.array([1]*n_action, dtype=np.float32), 
                                       shape=(n_action,)) 
        
        # Define max and min observation to reduce training time 
        ...
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        # Environment setting
        fixTargetMovingDirection = 1
        denseReward = 1
        ...
        
        # Movement of the robot arm 
        ...
    
        # Observation
        ...
        
        # Reward
        ...
            
        # Collision check for safty rate calculation   
        ...
        return np.array(self.observation).astype(np.float32), reward, done, info, accuracy, safe_rate
                                            
    def reset(self):
        ...
        return np.array(self.observation).astype(np.float32)
            
    def render(self, mode='human'):
        return

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()