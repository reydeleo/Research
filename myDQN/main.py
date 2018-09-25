import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import gym
import skimage as skimage
from skimage import transform, color, exposure
from itertools import count
import collections
import random


REPLAY_MEMORY_FILL_SIZE = 32#50000
REPLAY_MEMORY = collections.deque(maxlen=1000000)
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.1
GAMMA = 0.99
ACTION_REPEAT = 4 #this is the same thing as frame per action
NUMBER_OF_STEPS = 1000000 #in the paper this is 15 million

def create_network():
    # The network receives a frame from the game, flattened into an array.
    imageIn = tf.placeholder(shape=[None, 84, 84, 4],
                             dtype=tf.float32)  # here the scalarInput tensor is being reshaped into the shape [-1,84,84,3] so what it means is that it is a batch of unknown size that is going to have 3 frames together which are 84*84
    conv1 = tf.layers.conv2d(inputs=imageIn, filters=32, kernel_size=8, strides=4, padding='VALID',
                             activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=4, strides=2, padding='VALID', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=3, strides=1, padding='VALID', activation=tf.nn.relu)
    flat = tf.layers.flatten(inputs=conv3)
    dense = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)
    qValues = tf.layers.dense(inputs=dense, units=4)

    return imageIn, qValues


def loss(networkQValues):
    computed_qvalues = tf.placeholder(shape=[None, 4], dtype=tf.float32)
    meanSquaredError = tf.reduce_mean(tf.square(computed_qvalues - networkQValues))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00025)
    minimized = optimizer.minimize(meanSquaredError)

    return minimized, computed_qvalues


def gameStartState(Env):
    obs = Env.reset()
    grayImage = skimage.color.rgb2gray(obs)
    grayImage = skimage.transform.resize(grayImage, (84, 84))
    frames = []
    for times in range(4):
        frames.append(grayImage)
    state = np.stack(frames, axis=-1).reshape((1, 84, 84, 4))

    return state, Env


def transform(image):
    grayImage = skimage.color.rgb2gray(image)
    grayImage = skimage.transform.resize(grayImage, (84, 84))
    grayImage = grayImage.reshape((1,84,84,1))
    return grayImage

def initiliazeReplayMemory(environmentUsed):
    state, environment = gameStartState(environmentUsed)
    for steps in range(REPLAY_MEMORY_FILL_SIZE):
        action = environment.action_space.sample()
        observation, reward, done,_ = environment.step(action)
        observation = transform(observation)
        environment.render()
        nextState = np.append(observation, state[:, :, :, :3], axis=3)  #this "state[:, :, :, :3]" means that I am getting the first 3 frames and discarding the last frame and the fact that I am doing append means that I am going to put the new frame in the front because that is how append works
        singleExperience = (state, nextState, reward, action, done)
        REPLAY_MEMORY.append(singleExperience)
        state = nextState
        if done:
            state,environment = gameStartState(environment)

def qValueFormula(rwd, gamma, maxQValueNextState):
    qVal = rwd + gamma*maxQValueNextState
    return qVal

def create_session(memory_fraction=0.1, allow_growth=True):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_growth
    config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    return tf.Session(config=config)

def updatingTargetNetwork():
    onlineNetworkWeights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="onlineNetwork")
    targetNetworkWeights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="targetNetwork")
    a = []
    for weight_online, weight_target in zip(onlineNetworkWeights, targetNetworkWeights):
        a.append(tf.assign(weight_target, weight_online))
    operator = tf.group(*a)
    return operator


def deepLearningAlgorithm():
    global INITIAL_EPSILON
    #making the environment
    env = gym.make('Breakout-v0')

    # initilize replay memory
    initiliazeReplayMemory(env)

    # initialize action-value function Q with random weights
    ''' This is for when there is a GPU that is available
        to use'''

    # with tf.device('/device:GPU:1'):
    #     imageIn, qValues = create_network()
    #     minimized, computed_qvalues = loss(qValues)
    #
    #     with tf.variable_scope("targetNetwork"):  #this is where I make the target network
    #         imageIn_T, qValues_T = create_network()
    #
    #     operator=updatingTargetNetwork()
    #     sess = create_session()
    #     sess.run(tf.global_variables_initializer())

    '''----------------------------------------------------'''

    ''' This is when theres no GPU'''
    with tf.variable_scope("onlineNetwork"):  #this is where I make the online network
        imageIn, qValues = create_network()
        minimized, computed_qvalues = loss(qValues)

    with tf.variable_scope("targetNetwork"):  #this is where I make the target network
        imageIn_T, qValues_T = create_network()

    operator = updatingTargetNetwork()
    sess = create_session()
    sess.run(tf.global_variables_initializer())

    totalNumEpisodes = count()
    totalNumSteps = 0
    flag = 0 #this is just to say that the number of steps is already complete and stop going through the making more episode loop
    while totalNumSteps < NUMBER_OF_STEPS:
        for episode in totalNumEpisodes:
            if flag == 1:
                break
            state,env = gameStartState(env) # initialize starting game state
            nextState = None
            totalRewardForOneEpisode = 0
            steps = 0  # this is to count the number of steps in each single episode
            while True:
                randomNum = random.random()
                if randomNum < INITIAL_EPSILON:
                    action = env.action_space.sample()
                elif randomNum > INITIAL_EPSILON:
                    values = sess.run(qValues, feed_dict={imageIn: state})
                    for i,value in enumerate(values):
                        print(str(i) +": " + str(value))
                    maxQValue = values.argmax()
                    action = maxQValue

                observation, reward, done, _ = env.step(action)
                for repeat in range(ACTION_REPEAT-1): #this is to make sure that the action is repeated 4 times as the paper says
                    _,rwa,done, _ = env.step(action)
                    if done:
                        break
                    reward += rwa
                observation = transform(observation)
                env.render()  # optional
                nextState = np.append(observation, state[:, :, :, :3], axis=3)
                singleExperience = (state, nextState, reward, action, done)
                REPLAY_MEMORY.append(singleExperience)
                state = nextState
                INITIAL_EPSILON -= 9e-7

                randomMinibatch = random.sample(REPLAY_MEMORY,32)
                for experience in randomMinibatch:
                    qValuesFromNextState = sess.run(qValues_T, feed_dict={imageIn_T:experience[1]})
                    maxQValueFromNextState = np.max(qValuesFromNextState)
                    newQValueForCurrentState = qValueFormula(experience[2], GAMMA, maxQValueFromNextState)
                    qValuesForCurrentState = sess.run(qValues, feed_dict={imageIn:experience[0]})
                    qValuesForCurrentState[0,experience[3]] = newQValueForCurrentState #here the q value of the action that was taken is given a better estimate of what its true q value is
                    sess.run(minimized, feed_dict={imageIn:experience[0], computed_qvalues: qValuesForCurrentState})

                if done: #if for the last single experience that was store done = True meaning that the episode ended
                    steps += 1
                    totalNumSteps += 1
                    print()
                    print("episode:" + str(episode) + " " + str(totalRewardForOneEpisode) + " " + str(steps))
                    # print(str(totalRewardForOneEpisode) + " " + str(steps))
                    print()

                    # with open("rewards.txt", "a") as file:
                    #     file.write("episode:" + str(episode) + " ")
                    #     file.write(str(totalRewardForOneEpisode) + " " + str(steps))
                    #     file.write("\n")

                    print("step:" + str(totalNumSteps))
                    break
                else:
                    steps += 1
                    totalRewardForOneEpisode += reward
                    totalNumSteps += 1
                    if totalNumSteps >= NUMBER_OF_STEPS:
                        flag = 1
                        break
                    if totalNumSteps % 10000 == 0:
                        sess.run(operator)








def main():
    deepLearningAlgorithm()


if __name__ == '__main__':
    main()




