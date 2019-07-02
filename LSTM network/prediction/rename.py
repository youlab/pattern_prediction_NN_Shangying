# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:06:21 2018

@author: YouLab
"""

import tensorflow as tf

def rename(checkpoint_dir,newname, dry_run=True):
    #checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            # Set the new name
            new_name = var_name
            if 'lstm_cell/biases' in new_name:
                new_name = new_name.replace("lstm_cell/biases", "lstm_cell/bias")
            if 'lstm_cell/weights' in new_name:
                new_name = new_name.replace("lstm_cell/weights", "lstm_cell/kernel")

            if dry_run:
                print('%s would be renamed to %s.' % (var_name, new_name))
            else:
                print('Renaming %s to %s.' % (var_name, new_name))
                # Rename the variable
                var = tf.Variable(var, name=new_name)

        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint_dir+newname)
            
model_dir = 'saved_model_13'+"/mymodel-499j3"
rename(model_dir,'_new', dry_run=False)