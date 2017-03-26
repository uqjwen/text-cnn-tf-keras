import tensorflow as tf
import numpy as np
from keras.layers import Embedding, Convolution1D, GlobalMaxPooling1D, Dense, Activation, Dropout

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name = "y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_prob")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            embedding_layer = Embedding(input_dim = vocab_size, output_dim  = embedding_size, input_length = sequence_length)
            self.embedded_chars = embedding_layer(self.input_x)


        pool_output = []
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-pool-{}".format(i)):
                conv1d = Convolution1D(nb_filter = num_filters,
                                        filter_length = filter_size,
                                        border_mode = 'valid',
                                        activation = 'relu')
                h1 = conv1d(self.embedded_chars)
                pooled = GlobalMaxPooling1D()(h1)
                pool_output.append(pooled)

        self.h_pool = tf.concat(1, pool_output)
        flat_hidden_dims = num_filters*len(filter_sizes)

        with tf.name_scope("dropout"):
            self.h_dropout = tf.nn.dropout(self.h_pool, self.dropout_keep_prob)

        with tf.name_scope("output"):

            full_connect = Dense(input_dim = flat_hidden_dims, units = num_classes)
            self.score = full_connect(self.h_dropout)



            self.prediction = tf.argmax(self.score,1)

            l2_loss += tf.nn.l2_loss(full_connect.trainable_weights[0])
            l2_loss += tf.nn.l2_loss(full_connect.trainable_weights[1])

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.score, labels = self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda*l2_loss

        with tf.name_scope("accuracy"):
            self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))