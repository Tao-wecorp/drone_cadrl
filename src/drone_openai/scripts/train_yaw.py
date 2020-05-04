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
    env_yaw = gym.make("Yaw-v0")
        
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('drone_openai')
    outdir = pkg_path + '/results'
    env_yaw = gym.wrappers.Monitor(env_yaw, outdir, force=True)

    Alpha = rospy.get_param("/alpha")
    Epsilon = rospy.get_param("/epsilon")
    Gamma = rospy.get_param("/gamma")
    epsilon_discount = rospy.get_param("/epsilon_discount")
    nepisodes = rospy.get_param("/nepisodes")
    nsteps = rospy.get_param("/nsteps")

    qlearning = QLearning(actions=range(env_yaw.action_space.n), alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearning.epsilon

    highest_reward = 0
    final_steps = numpy.ndarray(0)

    for x in range(nepisodes):
        rospy.loginfo ("STARTING Episode #"+str(x))
        
        cumulated_reward = 0  
        done = False
        if qlearning.epsilon > 0.05:
            qlearning.epsilon *= epsilon_discount
        
        observation = env_yaw.reset()
        state = ''.join(map(str, observation))
        
        for i in range(nsteps):
            action = qlearning.chooseAction(state)
            observation, reward, done, info = env_yaw.step(action)
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward
            print(highest_reward)
            
            nextState = ''.join(map(str, observation))
            qlearning.learn(state, action, reward, nextState)

            if not(done):
                state = nextState
            else:
                rospy.loginfo ("DONE")
                final_steps = numpy.append(final_steps, [int(i + 1)])
                break 

    env_yaw.close()
