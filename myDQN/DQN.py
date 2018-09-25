# import numpy as np
# import random
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import gym
# import scipy.misc
# import os

# def to_grayscale(img):
#     return np.mean(img, axis=2).astype(np.uint8)

# def downsample(img):
#     return img[::2, ::2]

# def preprocess(img):
#     return to_grayscale(downsample(img))


# env = gym.make('Pong-v0')
# env.reset()
# obs,_,_,_ = env.step(env.action_space.sample())
# newImage = to_grayscale(obs)
# print("what")
# plt.show()
if __name__ == '__main__':
print("Hello")








# class Qnetwork():
#     def __init__(self):
#         # The network receives a frame from the game, flattened into an array.
#         # It then resizes it and processes it through four convolutional layers.
#         self.scalarInput = tf.placeholder(shape=[None,21168],dtype=tf.float32)  #tf.placeholder creates a tensor that is going to be constantly fed
#         self.imageIn = tf.reshape(self.scalarInput, shape=[-1,84,84,3])  #here the scalarInput tensor is being reshaped into the shape [-1,84,84,3] so what it means is that it is a batch of unknown size that is going to have 3 frames together which are 84*84
#         self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8,8], stride=[4,4], padding='VALID')
#         self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[4,4], stride=[2,2], padding='VALID')








