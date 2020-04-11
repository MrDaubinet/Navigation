from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt
from info import Info
from dqn_agent import Agent
# Change File Name to the path/to/Banana.exe
env = UnityEnvironment(file_name="C:/Udacity/Deep Reinforcement Learning/deep-reinforcement-learning/p1_navigation/Banana_Windows_x86_64/Banana.exe")
# get the default brain


brain_name = env.brain_names[0]
brain = env.brains[brain_name]

info = Info(env, brain_name, brain)
# print out information
info.print_info()
# set action and state
action_size, state_size = info.getInfo()

# prepare for training
env_info = env.reset(train_mode=False)[brain_name] # reset the environment

# Create the agent
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

# test best agent
agent.test(env, brain_name)