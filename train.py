#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
# from text_cnn import TextCNN
from text_cnn_my import TextCNN
from tensorflow.contrib import learn
from data_helpers import Data_Loader
import sys

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
data_loader = Data_Loader(FLAGS.batch_size)




with tf.Graph().as_default():

    with tf.Session() as sess :

        # Define Training procedure
        # print cnn.loss
        cnn = TextCNN(
            sequence_length=data_loader.sequence_length,
            num_classes=data_loader.num_classes,
            vocab_size=data_loader.vocab_size,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)






        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        checkpoint_dir = './checkpoint/'
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print (" [*] Loading parameters success...")
        else:
            print (" [!] Loading parameters failed!!!")

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            sys.stdout.write("\r{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            sys.stdout.flush()
            # train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, loss, accuracy = sess.run(
                [global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, loss, accuracy))

        for e in range(FLAGS.num_epochs):
            data_loader.reset_pointer()
            total_batches = data_loader.total_batches

            for b in range(total_batches):
                x_batch,y_batch = data_loader.next_batch()
                train_step(x_batch, y_batch)
                if (e*total_batches+b)%FLAGS.checkpoint_every == 0 or\
                    (e==FLAGS.num_epochs-1 and b == total_batches-1):
                        saver.save(sess, checkpoint_dir+'model.ckpt', global_step = e*total_batches+b)
                        dev_step(data_loader.x_dev, data_loader.y_dev)