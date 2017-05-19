# -*- coding: utf-8 -*-
"""
Created on Fri May 19 18:25:00 2017

@author: lin chen

a simple examples for the usage of tf.nn.top_k function
"""

import tensorflow as tf
import numpy as np


nnn=tf.placeholder(tf.float32,shape = [128])

mm = tf.nn.top_k(nnn, 3)

xxx = np.random.rand(128)
sess = tf.Session()
yyy = sess.run(mm,feed_dict = {nnn:xxx})
# now accessing the values and indices of the retrieved vectors
print(yyy.values)
print(yyy.indices)