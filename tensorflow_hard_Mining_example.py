# -*- coding: utf-8 -*-
"""
Created on Sat May 20 21:59:04 2017

@author: lin chen
example for using hard mining in training
"""

import tensorflow as tf
import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


x_image = tf.placeholder(tf.float32, [128,28,28,1])
y_ = tf.placeholder(tf.float32, shape=[128, 10])
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
# which is 
cross_entropy_each = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
# gather only the 30 samples with biggest loss
cross_entropy_hardmined_each = tf.gather(cross_entropy_each, tf.nn.top_k(cross_entropy_each,k = 30).indices)
#cross_entropy_hardmined =  tf.nn.top_k(cross_entropy_each,30)
cross_entropy_hardmined = tf.reduce_mean(cross_entropy_hardmined_each)
cross_entropy = tf.reduce_mean(cross_entropy_each)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step_hardmine = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_hardmined)
input_image = np.random.rand(128,28,28,1)
labels = np.random.rand(128,10)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
#fist do normal updating
for i in range(20):
    train_step.run(feed_dict={x_image:input_image, y_:labels})
    print("          step ",i,":", cross_entropy.eval(feed_dict={x_image:input_image, y_:labels}))
# then do hard-mined updating
for i in range(20):
    train_step_hardmine.run(feed_dict={x_image:input_image, y_:labels})
    print("hard_mine step ",i,":", cross_entropy.eval(feed_dict={x_image:input_image, y_:labels}))