# -*- coding: utf-8 -*-

#there are four fundamental concepts which underpin the RL
'''
Agent: the actor operating within the environment, it usually governed by a policy( rules)
Environment: the world in which the agent can operate in
Action: the agent can do something in the environment known as action
Reward and Observation: in return the agent receives a reward and a view of what the environment
looks like after acting on it.

OpenAI Gym is an environment which supports the following spaces

Box - n dimensional tensor, range of values,
e.g. Box(0,1,shape=(3,3))

Discrete - set of items 
e.g. Discrete(3)

Tuple -tuple of other spaces
e.g Tuple((Discrete(2),Box(0,100,shape=(1,))))

Dict - dictionary of spaces
e.g. Dict({'height':Discrete(2),'speed':Box(0,100,shape=(1,))})

MultiBinary - one hot encoded binary values
e.g. MultiBinary(4)

MultiDiscrete - multiple discrete values
e.g. MultiDiscrete([5,2,2])




to install stable_baselines 
!pip install stable_baselines3[extra]
'''
import os
#env
import gym
#ALGORITHM
from stable_baselines3 import PPO
#stable baseline lets you run 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


#load environment
environment_name='CartPole-v0'
env=gym.make(environment_name)
#lets test our environment
#testing cartpole env 5 times
episodes = 5
for episode in range(1, episodes+1):
    #set of observations
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        #view graphical representation of the envs, its does not work in acolab 
        env.render()
        #gives a sample of the action space, space may discrete(2), which means response is either 0 or 1
        #to show action space, env.action_space
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()




























