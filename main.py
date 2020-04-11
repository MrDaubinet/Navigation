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

# train the agent
scores = agent.dqn(
    env=env,
    brain_name=brain_name,
    n_episodes=2000, 
    max_t=1000, 
    eps_start=1.0, 
    eps_end=0.01, 
    eps_decay=0.995
)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# test best agent
score = agent.test(env, brain_name)
print("Score: {}".format(score))

