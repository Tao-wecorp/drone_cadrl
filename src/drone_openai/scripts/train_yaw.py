#!/usr/bin/env python

import numpy
import random
import time

import rospy
import rospkg

import env_yaw
import gym
from gym.spaces import *
from helpers.qlearning import QLearning


if __name__ == '__main__':
    rospy.init_node('train_node', anonymous=True)
    env = gym.make("Yaw-v0")

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    epsilon_discount = 0.99
    episodes = 1000
    steps = 200

    qlearning = QLearning(actions=range(env.action_space.n), alpha=alpha, gamma=gamma, epsilon=epsilon)
    initial_epsilon = qlearning.epsilon
    highest_reward = 0
    final_steps = numpy.ndarray(0)

    for x in range(episodes):
        rospy.loginfo ("STARTING Episode #"+str(x))
        
        cumulated_reward = 0  
        done = False
        if qlearning.epsilon > 0.05:
            qlearning.epsilon *= epsilon_discount
        
        observation = env.reset()
        state = ''.join(map(str, observation))
        
        for i in range(steps):
            action = qlearning.chooseAction(state)
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward
            
            nextState = ''.join(map(str, observation))
            qlearning.learn(state, action, reward, nextState)

            if not(done):
                state = nextState
            else:
                rospy.loginfo ("DONE")
                final_steps = numpy.append(final_steps, [int(i + 1)])
                break 

    env.close()
