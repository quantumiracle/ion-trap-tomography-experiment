#train 2 branches with different times

from __future__ import print_function
import tensorflow as tf

import numpy as np
import random
import cPickle as pickle
import matplotlib.pyplot as plt
import argparse
import math
save_file='./model.ckpt'

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)


args = parser.parse_args()
#print(tf.reduce_sum([[1,2],[3,4]],reduction_indices=[0,1]))

def compute_accuracy(v_xs, v_ys, v_p_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    p_y_pre = sess.run(p_prediction, feed_dict={xs: v_xs, keep_prob: 1})
    #print(y_pre,v_ys)
    error_og=tf.reduce_sum((abs(y_pre-v_ys)))
    error_p=tf.reduce_sum((abs(p_y_pre-v_p_ys)))
    
    result1 = sess.run(error_og, feed_dict={xs: v_xs, ys: v_ys, p_ys: v_p_ys, keep_prob: 1})
    result2 = sess.run(error_p, feed_dict={xs: v_xs, ys: v_ys, p_ys: v_p_ys, keep_prob: 1})
    return result1,result2

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,1,1], strides=[1,2,1,1], padding='SAME')

def leakyrelu(x, alpha=0.3, max_value=None):  #alpha need set
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

#############################
#general parameters
num_points=200
t=np.linspace(0,400,num_points)*1e-6   #time range, num of time points
om0=8.0e4   #om: 10k-100k(0-100k)
gamma0=2.0e-2  #gamma: 0-10k
num_p=4  #num of states distributions
sum_y=[]

###############################
#nn structure

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, num_points])   # 28x28
ys = tf.placeholder(tf.float32, [None, 2])  #num_p add 1 om
p_ys = tf.placeholder(tf.float32, [None, num_p])
W_conv1 = weight_variable([5,1, 1,32]) # patch 5x1, in size 1, out size 32; [a,b,c,d]: a,b,c equals to length, width, height(usually caused by num of convolution cores or inputs with height like RGB)
b_conv1 = bias_variable([32])
W_conv2 = weight_variable([5,1, 32, 64]) # patch 5x1, in size 32, out size 64
b_conv2 = bias_variable([64])


p_W_conv1 = weight_variable([5,1, 1,32]) # patch 5x1, in size 1, out size 32; [a,b,c,d]: a,b,c equals to length, width, height(usually caused by num of convolution cores or inputs with height like RGB)
p_b_conv1 = bias_variable([32])
p_W_conv2 = weight_variable([5,1, 32, 64]) # patch 5x1, in size 32, out size 64
p_b_conv2 = bias_variable([64])


#merge nn
#om nn
W_fc1 = weight_variable([num_points/4*64, 1024])
b_fc1 = bias_variable([1024])
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])


#p nn
p_W_fc1 = weight_variable([num_points/4*64, 512])
p_b_fc1 = bias_variable([512])
p_W_fc2 = weight_variable([512, num_p])
p_b_fc2 = bias_variable([num_p])


saver = tf.train.Saver()  #define saver of the check point
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, num_points, 1, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]








#mutual cnn
## conv1 layer ##

h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
#h_conv1 = leakyrelu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)                    # output size 14x14x32

## conv2 layer ##

h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
#h_conv2 = leakyrelu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64


p_h_conv1 = tf.nn.tanh(conv2d(x_image, p_W_conv1) + p_b_conv1) # output size 28x28x32
#h_conv1 = leakyrelu(conv2d(x_image, W_conv1) + b_conv1)
p_h_pool1 = max_pool_2x2(p_h_conv1)                    # output size 14x14x32

## conv2 layer ##

p_h_conv2 = tf.nn.tanh(conv2d(p_h_pool1, p_W_conv2) + p_b_conv2) # output size 14x14x64
#h_conv2 = leakyrelu(conv2d(h_pool1, W_conv2) + b_conv2)
p_h_pool2 = max_pool_2x2(p_h_conv2)                                         # output size 7x7x64




#om nn
## fc1 layer ##

# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, num_points/4*64])
h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#h_fc1 = leakyrelu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##



prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2



#p nn
p_h_pool2_flat = tf.reshape(p_h_pool2, [-1, num_points/4*64])
p_h_fc1 = tf.nn.tanh(tf.matmul(p_h_pool2_flat, p_W_fc1) + p_b_fc1)
#p_h_fc1 = leakyrelu(tf.matmul(p_h_pool2_flat, p_W_fc1) + p_b_fc1)
p_h_fc1_drop = tf.nn.dropout(p_h_fc1, keep_prob)



p_prediction = tf.nn.softmax(tf.matmul(p_h_fc1_drop, p_W_fc2) + p_b_fc2)




lr = tf.placeholder(tf.float32)

loss = tf.reduce_mean(tf.reduce_sum(abs(ys - prediction),
                                        reduction_indices=[1]))       # loss
#train_step = tf.train.AdamOptimizer(lr1).minimize(loss)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

p_loss = tf.reduce_mean(-tf.reduce_sum(p_ys * tf.log(p_prediction),
                                              reduction_indices=[1])) 

p_train_step = tf.train.AdamOptimizer(lr).minimize(p_loss)
    




sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)




#print(test_ys)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

persize_train=10  #num of samples per step in train set 

fx=open('x.pkl','rb')
fy1=open('y.pkl','rb')
fy2=open('yp.pkl','rb')
pick_size = 5000
test_size = 10 
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

if args.train:

    #generate test set
    batch_xs=[]
    batch_ys=[]
    batch_p_ys=[]
    test_xs=[]
    test_ys=[]
    test_p_ys=[]
    #test_size=50

    fig_x=[]
    fig_y=[]
    fig_y_p=[]
    try:
            batch_xs=pickle.load(fx)
            batch_ys=pickle.load(fy1)
            batch_p_ys=pickle.load(fy2)
    except Exception, e:
        print(e)
    test_xs=batch_xs[:test_size]
    test_ys=batch_ys[:test_size]
    test_p_ys=batch_p_ys[:test_size]
    if len(test_ys) > 100:
        print('error test size: ', len(test_ys))
    j = 0
    for i in range(10000):
        #i_xs=train_xs[i*persize_train/5:i*persize_train/5+persize_train]
        #i_ys=train_ys[i*persize_train/5:i*persize_train/5+persize_train]
        
        #load data from .pkl
        
        if (i*persize_train) % pick_size == 0:
            train_xs=[]
            train_ys=[]
            train_p_ys=[]
            train_xs = pickle.load(fx)
            train_ys = pickle.load(fy1)
            train_p_ys = pickle.load(fy2)
            j = 0
            if len(train_ys) < 5000:
                print('error train size: ', len(train_ys))
            
        if j <= len(train_ys)/persize_train-1:
            i_xs=train_xs[j*persize_train:j*persize_train+persize_train]
            i_ys=train_ys[j*persize_train:j*persize_train+persize_train]
            if j*persize_train + persize_train > pick_size:
                print('error: lack of train data', i) 
            i_p_ys=train_p_ys[j*persize_train:j*persize_train+persize_train]
            #if j % 3 == 0:    #5times p, 1 times ome+gamma
                #sess.run(train_step, feed_dict={xs: i_xs, ys: i_ys, keep_prob: 0.5})
            if i < 4000:
                sess.run(p_train_step, feed_dict={xs: i_xs, p_ys:i_p_ys, keep_prob: 0.5, lr:0.0001})
            elif i < 8000:
                sess.run(p_train_step, feed_dict={xs: i_xs, p_ys:i_p_ys, keep_prob: 0.5, lr:0.00001})
            else:
                sess.run(p_train_step, feed_dict={xs: i_xs, p_ys:i_p_ys, keep_prob: 0.5, lr:0.000005})
            j=j+1
            #print(i_ys)
        if i % 5 == 0:
            error_og, error_p=compute_accuracy(test_xs,test_ys,test_p_ys)
            print(error_og, error_p)
            ##ways of plot
            #1.
            #plt.plot(i,error)
            #2.
            #ax.scatter(i, error)
            #3.
            fig_x.append(i)
            fig_y.append(error_og)
            fig_y_p.append(error_p)
            #plt.pause(1)
            plt.plot(fig_x, fig_y, color='blue')
            plt.plot(fig_x, fig_y_p, color='red')
            plt.ylim(0,50)
            #plt.pause(0.1)
            
            #plt.close()
    saver.save(sess, save_file)
    plt.savefig('com.png')
    plt.show()
    



'''
    sess.run(init)
    fig_x=[]
    fig_y=[]
    j = 0
    for i in range(4000):
        #i_xs=train_xs[i*persize_train/5:i*persize_train/5+persize_train]
        #i_ys=train_ys[i*persize_train/5:i*persize_train/5+persize_train]
        
        #load data from .pkl
        
        if (i*persize_train) % pick_size == 0:
            train_xs=[]
            train_ys=[]
            train_p_ys=[]
            train_xs = pickle.load(fx)
            train_ys = pickle.load(fy1)
            train_p_ys = pickle.load(fy2)
            j = 0
            if len(train_ys) < 5000:
                print('error train size: ', len(train_ys))
            
        if j <= len(train_ys)/persize_train-1:
            i_xs=train_xs[j*persize_train:j*persize_train+persize_train]
            i_ys=train_ys[j*persize_train:j*persize_train+persize_train]
            if j*persize_train + persize_train > pick_size:
                print('error: lack of train data', i) 
            i_p_ys=train_p_ys[j*persize_train:j*persize_train+persize_train]
            sess.run(train_step2, feed_dict={xs: i_xs, ys: i_ys, keep_prob: 0.5})
            sess.run(p_train_step2, feed_dict={xs: i_xs, p_ys:i_p_ys, keep_prob: 0.5})
            j=j+1
            #print(i_ys)
        if i % 5 == 0:
            error=compute_accuracy(test_xs,test_ys,test_p_ys)
            print(error)
            ##ways of plot
            #1.
            #plt.plot(i,error)
            #2.
            #ax.scatter(i, error)
            #3.
            fig_x.append(i)
            fig_y.append(error)
            #plt.pause(1)
            plt.plot(fig_x, fig_y, color='red')
            plt.ylim(0,50)
            plt.pause(0.1)
            plt.savefig('com.png')
    saver.save(sess, save_file)
   ''' 

if args.test:
    #test data generate
    test_xs=[]
    test_ys=[]
    test_p_ys=[]
    test_size=100
    for j in range(test_size):  #num of samples
        p=[]  #initialize outside the for loop and del inside the loop would be wrong
        om_r_set=[]
        sum_p=0
        sum_y=0  #final wave
        om = 1
        gamma = 1
        '''
        om=random.uniform(0.8,1.2)
        gamma=random.uniform(0.8,1.2)
        '''
        #print(om)
        for i in range (0,num_p):
            p_i=random.uniform(0,10)
            p.append(p_i)
            sum_p=sum_p+p_i
        om_r_set.append(om) #normalized om(0-1) ~p
        om_r_set.append(gamma)
        
        for i in range (0,num_p):
            p[i]=p[i]/sum_p     #normalize sum of p[i] to be 1
            Om=om*om0*math.sqrt(i+1)
            Gamma=gamma*gamma0*(i+1)**0.7
            y=p[i]*np.cos(2*Om*t)*np.exp(-Gamma*t)
            #om_i=math.sqrt(i) #!!!
            #om_i=random.uniform(0,10)   # random omega
            #y=p[i]*np.cos(om_i*om*om0*t)*np.cos(om_i*om*om0*t)*np.exp(-gamma*gamma0*t)
            sum_y+=y
        #sum_y=np.fft.fft(sum_y)   #fft
        sum_y -= np.mean(sum_y, axis=0)  #zero-center
        sum_y /= np.std(sum_y, axis=0)   #normalize
        test_xs.append(sum_y)
        test_ys.append(om_r_set)
        test_p_ys.append(p)
        if j % 10000 == 0:
            print(j)

    saver.restore(sess, save_file)
    #print(test_ys[:5],test_p_ys[:5])
    y_pre_sample = sess.run(prediction, feed_dict={xs: test_xs[:5], keep_prob: 1})
    p_y_pre_sample = sess.run(p_prediction, feed_dict={xs: test_xs[:5], keep_prob: 1})
    #sample_acc=compute_accuracy(test_xs[:5],test_ys[:5], test_p_ys[:5])
    for i in range (5):
        print(test_ys[i],'\n',y_pre_sample[i],';')
    for i in range (5):
        print(test_p_ys[i],'\n',p_y_pre_sample[i],';')
    #print(y_pre_sample,p_y_pre_sample,sample_acc)
