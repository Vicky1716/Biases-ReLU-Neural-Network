
'''
@ Summary: Biases ReLU Neural Network (Tensorflow Version)
@ Author: XL.Liang, ZY.Luo
'''
from matplotlib.pyplot import get
import tensorflow as tf
import time
import numpy as np
BATCH_SIZE = 32
import load_data
import pandas as pd
import random
tf.reset_default_graph()
np.set_printoptions(threshold=np.inf)
scale = 0.01
def batch_norm_layer(x, train_phase, scope_bn):
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(0.0)
        gamma = tf.Variable(1.0)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed,beta,gamma,mean,var

def norm_layer(x,train_phase,scope_bn):
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(0.0)
        gamma = tf.Variable(1.0)

        mean = tf.reduce_mean(x)
        length = BATCH_SIZE
        var = tf.sqrt((tf.reduce_sum((x-mean)**2))/length)
        normed = (x-mean)/var
    return normed,beta,gamma,mean,var

def get_data():
    datasets=load_data.read_data_sets(tf.float32,"data//cout_file_1.csv")
    with tf.name_scope('input'):
        xs = tf.placeholder(tf.float32, [None, 12], name='x_input')
        ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
    return xs, ys, datasets

def Urelu_layer(inputs,scope_bn,istrain):
    normed,beta,gamma,mean1,var1=norm_layer(inputs, istrain, scope_bn)
    normed1=normed
    outputs1 = tf.maximum(0.0, normed1)

    quantile=[-3*var1+mean1,-0.834*var1+mean1,-0.248*var1+mean1,0.248*var1+mean1,0.834*var1+mean1]
    for i in range(len(quantile)):
        output = tf.maximum(0.0, normed1 - quantile[i]*gamma+beta)
        outputs1 = tf.concat([outputs1, output], 1)
    return outputs1,normed,gamma,beta
