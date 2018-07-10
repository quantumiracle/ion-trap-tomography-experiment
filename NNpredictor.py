from __future__ import print_function
import tensorflow as tf

import numpy as np
import random
import cPickle as pickle
import matplotlib.pyplot as plt
import argparse
import math


class Predictor(object):
    '''
    def __init__(self, curve_set):
        self.curve_set = curve_set
    '''
    def compute_accuracy(self, v_xs, v_ys, v_p_ys):
        global prediction
        y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
        p_y_pre = sess.run(p_prediction, feed_dict={xs: v_xs, keep_prob: 1})
        #print(y_pre,v_ys)
        error_og=tf.reduce_sum((abs(y_pre-v_ys)))
        error_p=tf.reduce_sum((abs(p_y_pre-v_p_ys)))
        
        result1 = sess.run(error_og, feed_dict={xs: v_xs, ys: v_ys, p_ys: v_p_ys, keep_prob: 1})
        result2 = sess.run(error_p, feed_dict={xs: v_xs, ys: v_ys, p_ys: v_p_ys, keep_prob: 1})
        return result1,result2

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        # stride [1, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] = 1
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1,2,1,1], strides=[1,2,1,1], padding='SAME')

    def leakyrelu(self, x, alpha=0.3, max_value=None):  #alpha need set
        '''ReLU.

        alpha: slope of negative section.
        '''
        negative_part = tf.nn.relu(-x)
        x = tf.nn.relu(x)
        if max_value is not None:
            x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                                tf.cast(max_value, dtype=tf.float32))
        x -= tf.constant(alpha, dtype=tf.float32) * negative_part
        return x
    
    def NNpredictor(self, dim, curve_set):
        #############################
        #general parameters
        num_points=200
        t=np.linspace(0,400,num_points)   #time range, num of time points
        om0=8.0e4   #om: 10k-100k(0-100k)
        gamma0=2.0e-2  #gamma: 0-10k
        num_p=dim  #num of states distributions
        sum_y=[]

        ###############################
        #nn structure

        # define placeholder for inputs to network
        xs = tf.placeholder(tf.float32, [None, num_points])   # 28x28
        ys = tf.placeholder(tf.float32, [None, 2])  #num_p add 1 om
        p_ys = tf.placeholder(tf.float32, [None, num_p])
        W_conv1 = self.weight_variable([5,1, 1,32]) # patch 5x1, in size 1, out size 32; [a,b,c,d]: a,b,c equals to length, width, height(usually caused by num of convolution cores or inputs with height like RGB)
        b_conv1 = self.bias_variable([32])
        W_conv2 = self.weight_variable([5,1, 32, 64]) # patch 5x1, in size 32, out size 64
        b_conv2 = self.bias_variable([64])


        p_W_conv1 = self.weight_variable([5,1, 1,32]) # patch 5x1, in size 1, out size 32; [a,b,c,d]: a,b,c equals to length, width, height(usually caused by num of convolution cores or inputs with height like RGB)
        p_b_conv1 = self.bias_variable([32])
        p_W_conv2 = self.weight_variable([5,1, 32, 64]) # patch 5x1, in size 32, out size 64
        p_b_conv2 = self.bias_variable([64])


        #merge nn
        #om nn
        W_fc1 = self.weight_variable([num_points/4*64, 1024])
        b_fc1 = self.bias_variable([1024])
        W_fc2 = self.weight_variable([1024, 2])
        b_fc2 = self.bias_variable([2])


        #p nn
        p_W_fc1 = self.weight_variable([num_points/4*64, 512])
        p_b_fc1 = self.bias_variable([512])
        p_W_fc2 = self.weight_variable([512, num_p])
        p_b_fc2 = self.bias_variable([num_p])


        saver = tf.train.Saver()  #define saver of the check point
        keep_prob = tf.placeholder(tf.float32)
        x_image = tf.reshape(xs, [-1, num_points, 1, 1])
        # print(x_image.shape)  # [n_samples, 28,28,1]








        #mutual cnn
        ## conv1 layer ##

        h_conv1 = tf.nn.tanh(self.conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
        #h_conv1 = self.leakyrelu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)                    # output size 14x14x32

        ## conv2 layer ##

        h_conv2 = tf.nn.tanh(self.conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
        #h_conv2 = self.leakyrelu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)                                         # output size 7x7x64


        p_h_conv1 = tf.nn.tanh(self.conv2d(x_image, p_W_conv1) + p_b_conv1) # output size 28x28x32
        #h_conv1 = self.leakyrelu(self.conv2d(x_image, W_conv1) + b_conv1)
        p_h_pool1 = self.max_pool_2x2(p_h_conv1)                    # output size 14x14x32

        ## conv2 layer ##

        p_h_conv2 = tf.nn.tanh(self.conv2d(p_h_pool1, p_W_conv2) + p_b_conv2) # output size 14x14x64
        #h_conv2 = self.leakyrelu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        p_h_pool2 = self.max_pool_2x2(p_h_conv2)                                         # output size 7x7x64




        #om nn
        ## fc1 layer ##

        # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
        h_pool2_flat = tf.reshape(h_pool2, [-1, num_points/4*64])
        h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        #h_fc1 = self.leakyrelu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        ## fc2 layer ##



        prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2



        #p nn
        p_h_pool2_flat = tf.reshape(p_h_pool2, [-1, num_points/4*64])
        p_h_fc1 = tf.nn.tanh(tf.matmul(p_h_pool2_flat, p_W_fc1) + p_b_fc1)
        #p_h_fc1 = self.leakyrelu(tf.matmul(p_h_pool2_flat, p_W_fc1) + p_b_fc1)
        p_h_fc1_drop = tf.nn.dropout(p_h_fc1, keep_prob)



        p_prediction = tf.nn.softmax(tf.matmul(p_h_fc1_drop, p_W_fc2) + p_b_fc2)




        lr1=0.00001
        lr2=0.000001
        loss = tf.reduce_mean(tf.reduce_sum(abs(ys - prediction),
                                                reduction_indices=[1]))       # loss
        #train_step = tf.train.AdamOptimizer(lr1).minimize(loss)
        train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

        p_loss = tf.reduce_mean(-tf.reduce_sum(p_ys * tf.log(p_prediction),
                                                    reduction_indices=[1])) 
        #p_train_step = tf.train.AdamOptimizer(0.000001).minimize(p_loss)
        p_train_step = tf.train.AdamOptimizer(0.0001).minimize(p_loss)

        #sess = tf.Session()
        with tf.Session() as sess:
            # important step
            # tf.initialize_all_variables() no long valid from
            # 2017-03-02 if using tensorflow >= 0.12
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
    
            sess.run(init)

            save_file='./model.ckpt'
            saver.restore(sess, save_file)
            P = sess.run(p_prediction, feed_dict={xs: curve_set, keep_prob: 1})
        return P

