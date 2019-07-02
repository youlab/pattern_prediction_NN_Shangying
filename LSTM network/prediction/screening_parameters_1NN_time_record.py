# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:40:08 2017

@author: fankai
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tcl
#import csv
import os
#import peakutils
#from peakutils.plot import plot as pplot
#from matplotlib import pyplot
import time
#import pdb
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

class Model(object):
    
    def __init__(self, train_mode=True, input_dim=13, T=150, prev=16,
                 lstm_size=256,
                 batch_size=100, e_learning_rate=1e-4,
                 ):
        self.train_mode = train_mode
        self.input_dim = input_dim
        self.T = T
        self.prev = prev

        self.enc_size = lstm_size        
        
        self.batch_size = batch_size
        self.e_learning_rate = e_learning_rate

        self._srng = np.random.RandomState(np.random.randint(1,2147462579))
        
        self.lstm_enc = tf.contrib.rnn.LSTMCell(self.enc_size, state_is_tuple=True)
        
        # initial state
        self.enc_state = self.lstm_enc.zero_state(self.batch_size, tf.float32)
        self.yss = [0] * self.T
        self.y_prev = 0.0
        self.e_loss = 0.0
        
        # build computation graph of model
        self.DO_SHARE=None
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_dim])
        self.ymax = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.T])
        
        xe = self.input_embedding(self.x)
        self.hs = self.height_model(xe)
        self.p_loss = tf.reduce_mean(tf.square(self.hs - self.ymax))
        self.e_loss += self.p_loss
        
        for t in range(self.T): # range(self.T): if using python3
            
            self.y_prev = self.get_yprev(t)
            h_enc, self.enc_state = self.encode(self.enc_state, tf.concat([xe, self.y_prev], 1))
            ylt = self.linear(h_enc)
            self.yss[t] = tf.sigmoid(ylt)
            y_true = tf.reshape(self.y[:,t], [-1, 1])
            self.e_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=ylt, labels=y_true))
             
            self.DO_SHARE = True

        self.e_vars = tf.trainable_variables()

        self.e_optimizer = tf.train.AdamOptimizer(self.e_learning_rate, beta1=0.5, beta2=0.999)
        e_grads = self.e_optimizer.compute_gradients(self.e_loss, self.e_vars)
        clip_e_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in e_grads if grad is not None]
        self.e_optimizer = self.e_optimizer.apply_gradients(clip_e_grads)
            
    
    def train(self, train_set, valid_set, maxEpoch=10):
        
         with tf.Session() as sess:
            
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            
            i = 0
            for epoch in range(maxEpoch): # range for python3
                
                for xtrain, ytrain, ptrain in self.data_loader(train_set, self.batch_size, shuffle=True):
                    ytrain = ytrain[:,::-1]
                    
                    _, Le, Lp, ys, h = sess.run([self.e_optimizer, self.e_loss, self.p_loss, self.yss, self.hs], 
                                     feed_dict={self.x: xtrain, self.y: ytrain, self.ymax: ptrain})
                    i += 1
                    
                    if i % 10 == 0:
                        Levs, Lpvs = [], []
                        for xvalid, yvalid, pvalid in self.data_loader(valid_set, self.batch_size):
                            yvalid = yvalid[:,::-1]
                            
                            Lev, Lpv, ysv, hv = sess.run([self.e_loss, self.p_loss, self.yss, self.hs], feed_dict={self.x: xvalid, self.y: yvalid, self.ymax: pvalid})
                            Levs.append(Lev)
                            Lpvs.append(Lpv)
                        Le_valid = np.array(Levs).mean()
                        Lp_valid = np.array(Lpvs).mean()
                        print("Iter=%d: Le: %f Lp: %f Le_valid: %f Lp_valid: %f" % (i, Le, Lp, Le_valid, Lp_valid))
                        #print(ys)
                        #print(h)

                
#                self.save_model(saver, sess, step=epoch)
#                np.savetxt('ys_epoch'+str(epoch)+'.txt', ys )
#                np.savetxt('ytrain_epoch'+str(epoch)+'.txt', ytrain )
#                np.savetxt('h_epoch'+str(epoch)+'.txt',h )
#                np.savetxt('ptrain_epoch'+str(epoch)+'.txt',ptrain )
#                np.savetxt('ysv_epoch'+str(epoch)+'.txt', ysv )
#                np.savetxt('yvalid_epoch'+str(epoch)+'.txt', yvalid )
#                np.savetxt('hv_epoch'+str(epoch)+'.txt',hv )
#                np.savetxt('pvalid_epoch'+str(epoch)+'.txt',pvalid )  
    
    def data_loader(self, train_set, batchsize, shuffle=False): 
        features, labels, peaks = train_set
        if shuffle:
            indices = np.arange(len(features))
            self._srng.shuffle(indices)
        for start_idx in range(0, len(features) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield features[excerpt], labels[excerpt], peaks[excerpt]
            
    def data_loader2(self, test_set, batchsize, shuffle=False): 
        for start_idx in range(0, len(test_set) - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield test_set[excerpt]

    
    
    def encode(self, state, input):
        """
        run LSTM
        state = previous encoder state
        input = cat(read,h_dec_prev)
        returns: (output, new_state)
        """
        with tf.variable_scope("e_lstm",reuse=self.DO_SHARE):
            return self.lstm_enc(input,state)
            
    #fully_connected creates a variable called weights,
    #representing a fully connected weight matrix, which is multiplied by the inputs to produce a Tensor of hidden units
    def linear(self, x):
        with tf.variable_scope("e_linear", reuse=self.DO_SHARE):
            yl = tcl.fully_connected(inputs=x, num_outputs=1, activation_fn=None)
        return yl # output logits w.r.t sigmoid
    
    def input_embedding(self, x):
        with tf.variable_scope("e_eblinear1", reuse=None):
            h1 = tcl.fully_connected(inputs=x, num_outputs=128, activation_fn=tf.nn.relu)
        with tf.variable_scope("e_eblinear2", reuse=None):
            h2 = tcl.fully_connected(inputs=x, num_outputs=64, activation_fn=tf.nn.relu)
        return h2
    
    def height_model(self, x):
        with tf.variable_scope("e_hlinear1", reuse=None):
            h1 = tcl.fully_connected(inputs=x, num_outputs=64, activation_fn=tf.nn.relu)
        with tf.variable_scope("e_hlinear2", reuse=None):
            h2 = tcl.fully_connected(inputs=x, num_outputs=1, activation_fn=None)
        return h2

    def get_yprev(self, t):
        with tf.variable_scope("e_yprev", reuse=self.DO_SHARE):
            yp_init = tf.get_variable('yp_init', [self.batch_size, self.prev], initializer=tf.constant_initializer(0.5))
        return yp_init if t == 0 else tf.concat([self.y_prev[:,1:], self.yss[t-1]], 1)
            
           

if __name__ == "__main__":
    
    bsize=100
    ceter_edge=25
    center=501
    length=2*center-1
    x_axis=np.linspace(0, length, num=length)
    thred_pt=0.1

    mymodel = Model(train_mode=False, input_dim=13, T=501, batch_size=bsize)
    saver = tf.train.Saver()
    
    sample_size=60000
    time_record=np.zeros((int(sample_size/100),1))
    #[ G4 alpha beta Kphi exp_phi alpha_c alpha_T alpha_L KC KT KP d_A domainR]
    G4=np.random.rand(sample_size,1)*(1.0-1e-5)+1e-5 #
    alpha=np.random.rand(sample_size,1)
    beta=np.random.rand(sample_size,1)
    Kphi=np.random.rand(sample_size,1)*(1.0-1e-5)+1e-5
    exp_phi=np.random.rand(sample_size,1)
    alpha_c=np.random.rand(sample_size,1)*(1.0-0.04)+0.04
    alpha_T=np.random.rand(sample_size,1)*(1.0-0.01)+0.01
    alpha_L=np.random.rand(sample_size,1)*(1.0-0.01)+0.01
    KC=np.random.rand(sample_size,1)*(1.0-0.2)+0.2
    KT=np.random.rand(sample_size,1)*(1.0-0.01)+0.01
    KP=np.random.rand(sample_size,1)*(1.0-0.01)+0.01
    d_A=np.random.rand(sample_size,1)*(1.0-0.025)+0.025
    domainR=np.random.rand(sample_size,1)*(1.0-1.0/3.0)+1.0/3.0
    
    test_set = np.concatenate((G4, alpha, beta, Kphi, exp_phi, alpha_c, alpha_T, alpha_L, KC, KT, KP, d_A, domainR), axis=1)
    
    
    with tf.Session() as sess:

        saver.restore(sess, "saved_model_ce"+"/mymodel-499j3_new") #        
        ysts1=np.array([], dtype=np.float32).reshape(center,0,1)
        hts1=np.array([], dtype=np.float32).reshape(0,1)
        ii=0
        tic=time.time()
        for xtest in Model.data_loader2(mymodel, test_set,bsize):
            #pdb.set_trace() 
            #print(np.shape(xtest))

            yst1, ht1 = sess.run([mymodel.yss, mymodel.hs], feed_dict={mymodel.x: xtest})
            time_record[ii]=time.time()-tic
            ii=ii+1
            #pdb.set_trace() 
            ysts1 = np.concatenate((ysts1, yst1), axis=1)
            hts1 = np.concatenate((hts1, ht1), axis=0)
#
    with open('time_record_NN.txt','wb') as fl:
        np.savetxt(fl, time_record)
#        toc=time.time()
#        elapse_time=toc-tic
#        print(elapse_time)











































