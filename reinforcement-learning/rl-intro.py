# -*- coding: utf-8 -*-
import gym
import numpy as np
#we use this environment
env= gym.make('MountainCar-v0')
#any time we have an environment, the first thing we do is to reset
env.reset()

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

LEARNING_RATE=0.1
DISCOUNT=0.95
EPISODES=25000

SHOW_EVERY= 2000

DISCRETE_OS_SIZE=[20]*len(env.observation_space.high)
discrete_os_win_size=(env.observation_space.high - env.observation_space.low)/ DISCRETE_OS_SIZE


epsilon =0.5
START_EPSILON_DECAYING=1
END_EPSILON_DECAYING = EPISODES //2

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table= np.random.uniform(low=-2,high=0,size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state= ( state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


#print(discrete_state)
#print(q_table[discrete_state])
#print(np.argmax(q_table[discrete_state]))

for episode in range(EPISODES):
    if episode % SHOW_EVERY ==0:
       print(episode)
       render= True
    else:
        render= False
    
    #env.reset() gives the most recent state
    discrete_state= get_discrete_state(env.reset())
    done=False
    
    while not done:
        if np.random.random() > epsilon:
            #np.argmax, returns the index of the max of the array
            action=np.argmax(q_table[discrete_state])
        else:
            action= np.random.randint(0,env.action_space.n)
        
        #to make the action work, new_state is the position and velocity
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        #print(new_state)
        if render:
            
            #to show the graphic
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q= q_table[discrete_state +(action, )]
            #calculating q values formula
            new_q= (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT *max_future_q)
            q_table[discrete_state + (action, )] =new_q
        elif new_state[0] >= env.goal_position:
            print(f'We made it on episode{episode}')
            q_table[discrete_state + (action, )] = 0
            
        discrete_state = new_discrete_state
        
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    
env.close()
        
        
        