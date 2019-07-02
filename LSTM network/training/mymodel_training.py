# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:40:08 2017
Author: Fankai
Modified by Shangying on June 23, 2017
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tcl
import csv
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"]="0"


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs_j1"
logdir = "{}/run-{}/".format(root_logdir, now)


class Model(object):
    
    def __init__(self, model_path, train_mode=True, input_dim=25, T=180, prev=16,
                 lstm_size=256,
                 batch_size=100, e_learning_rate=1e-4,
                 ):
        self.model_path = model_path
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
        self.ys = [0] * self.T
        self.y_prev = 0.0
        self.e_loss = 0.0
        
        # build computation graph of model
        self.DO_SHARE=None
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim])
        self.ymax = tf.placeholder(tf.float32, shape=[None, 1])
        self.y = tf.placeholder(tf.float32, shape=[None, self.T])
        
        xe = self.input_embedding(self.x)
        self.h = self.height_model(xe)
        self.p_loss = tf.reduce_mean(tf.square(self.h - self.ymax))
        self.e_loss += self.p_loss
        
        for t in range(self.T): # range(self.T): if using python3
            
            self.y_prev = self.get_yprev(t)
            h_enc, self.enc_state = self.encode(self.enc_state, tf.concat([xe, self.y_prev], 1))
            ylt = self.linear(h_enc)
            self.ys[t] = tf.sigmoid(ylt)
            y_true = tf.reshape(self.y[:,t], [-1, 1])
            self.e_loss +=tf.reduce_mean(tf.square(y_true - self.ys[t])) #MSE
             
            self.DO_SHARE = True

        self.e_vars = tf.trainable_variables()

        self.e_optimizer = tf.train.AdamOptimizer(self.e_learning_rate, beta1=0.5, beta2=0.999)
        e_grads = self.e_optimizer.compute_gradients(self.e_loss, self.e_vars)
        clip_e_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in e_grads if grad is not None]
        self.e_optimizer = self.e_optimizer.apply_gradients(clip_e_grads)


        self.eloss_summary = tf.summary.scalar('eloss', self.e_loss)
        self.ploss_summary=tf.summary.scalar('ploss',self.p_loss)
        self.file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())



        """
        if self.train_mode == False:
            self.sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(self.sess, self.model_path + "/mymodel-10") #
            # code to load test data
            xtest =
            ytest =
            ptest =
            ys, h = self.sess.run([self.ys, self.h], feed_dict={self.x: xtest, self.y: ytest, self.ymax: ptest})
            # code save result
            self.sess.close()
        """
    
    def train(self, train_set, valid_set, maxEpoch=10):
        
         with tf.Session() as sess:
            
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            
            i = 0
            loss_v=np.zeros((maxEpoch,8))
            for epoch in range(maxEpoch): # range for python3
                Lds, Lps = [], []
                Ldvs, Lpvs = [], []                
                for xtrain, ytrain, ptrain in self.data_loader(train_set, self.batch_size, shuffle=True):
                    ytrain = ytrain[:,::-1]
                    
                    _, Le, Lp, ys, h = sess.run([self.e_optimizer, self.e_loss, self.p_loss, self.ys, self.h], 
                                     feed_dict={self.x: xtrain, self.y: ytrain, self.ymax: ptrain})
                    Ld=Le-Lp
                    Lds.append(Ld)
                    Lps.append(Lp)
                    i += 1
                    
                    if i % 1000 == 0:
                        summary_str_e = self.eloss_summary.eval(feed_dict={self.x: xtrain, self.y: ytrain, self.ymax: ptrain})
                        summary_str_p = self.ploss_summary.eval(feed_dict={self.x: xtrain, self.y: ytrain, self.ymax: ptrain})
                        self.file_writer.add_summary(summary_str_e, i)
                        self.file_writer.add_summary(summary_str_p, i)
                for xvalid, yvalid, pvalid in self.data_loader(valid_set, self.batch_size):
                    yvalid = yvalid[:,::-1]
                    
                    Lev, Lpv, ysv, hv = sess.run([self.e_loss, self.p_loss, self.ys, self.h], feed_dict={self.x: xvalid, self.y: yvalid, self.ymax: pvalid})
                    Ldv=Lev-Lpv
                    Ldvs.append(Ldv)
                    Lpvs.append(Lpv)
                Ld_train_mean = np.array(Lds).mean()
                Lp_train_mean = np.array(Lps).mean()
                Ld_valid_mean = np.array(Ldvs).mean()
                Lp_valid_mean = np.array(Lpvs).mean()
                Ld_train_std = np.array(Lds).std()
                Lp_train_std = np.array(Lps).std()
                Ld_valid_std = np.array(Ldvs).std()
                Lp_valid_std = np.array(Lpvs).std()
#                pdb.set_trace()
                loss_v[epoch,:]=[Ld_train_mean, Lp_train_mean, Ld_valid_mean, Lp_valid_mean, Ld_train_std, Lp_train_std, Ld_valid_std, Lp_valid_std]                

              
                self.save_model(saver, sess, step=epoch)
            np.savetxt('save_j1/ys.txt', ys )
            np.savetxt('save_j1/ytrain.txt', ytrain )
            np.savetxt('save_j1/h.txt',h )
            np.savetxt('save_j1/ptrain.txt',ptrain )
            np.savetxt('save_j1/ysv.txt', ysv )
            np.savetxt('save_j1/yvalid.txt', yvalid )
            np.savetxt('save_j1/hv.txt',hv )
            np.savetxt('save_j1/pvalid.txt',pvalid )  
            np.savetxt('save_j1/loss.txt', loss_v )
            self.file_writer.close()
    
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
        
    
    def save_model(self, saver, sess, step):
        """
        save model with path error checking
        """
        if self.model_path is None:
            my_path = "save" # default path in tensorflow saveV2 format
            # try to make directory if "save" path does not exist
            if not os.path.exists("save"):
                try:
                    os.makedirs("save")
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
        else: 
            my_path = self.model_path + "/mymodel"
                
        saver.save(sess, my_path, global_step=step)
    
    
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
        return yp_init if t == 0 else tf.concat([self.y_prev[:,1:], self.ys[t-1]], 1)
            

if __name__ == "__main__":
    
    # TODO: preprocessing dataset
    # Load data from csv file
    with open('all_data.csv') as csvfile1:
        mpg = list(csv.reader(csvfile1))
        results=np.array(mpg).astype("float")
        
    #assign 1500 data set to train set and the rest to valid set
        #data structure: 0:25 input parameters; 25:205 normalized output; end: peak value of the output
    

    train_size=int(len(results*0.8)/100)*100
    valid_size=int(len(results*0.1)/100)*100
    
    train_set = results[:train_size,0:13], results[:train_size,13:514], results[:train_size,-2:-1] #parameters,distribution,peak value
    valid_set = results[-valid_size:,0:13], results[-valid_size:,13:514], results[-valid_size:,-2:-1]


    train_mode = True
    mymodel = Model("saved_model", train_mode=train_mode, input_dim=13, T=501)
    if train_mode == True:
        mymodel.train(train_set, valid_set, maxEpoch=500) # # of iters = maxepoch * N/bs
            
