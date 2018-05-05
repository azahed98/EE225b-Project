"""Essentially the same as 
https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation/tree/master/data
but with some changes to make it easy to alter
"""

import os
import random
import tensorflow as tf
import time
import wget
import tarfile
import numpy as np
# import cv2

class CoCNN:
    def __init__(self, device='/gpu:0', checkpoint_dir='./checkpoints/', NUM_CATEGORIES=18):
        self.device = device
        # self.maybe_download_and_extract()
        self.NUM_CATEGORIES = NUM_CATEGORIES
        self.buildGraph()

        self.saver = tf.train.Saver(max_to_keep = 5, keep_checkpoint_every_n_hours =1)
        config = tf.ConfigProto(allow_soft_placement = True)
        self.session = tf.Session(config = config)
        self.session.run(tf.global_variables_initializer())
        self.checkpoint_dir = checkpoint_dir

    def maybe_download_and_extract(self):
        """Download and unpack VOC data if data folder only contains the .gitignore file"""
        if os.listdir('data') == ['.gitignore']:
            filenames = ['VOC_OBJECT.tar.gz', 'VOC2012_SEG_AUG.tar.gz', 'stage_1_train_imgset.tar.gz', 'stage_2_train_imgset.tar.gz']
            url = 'http://cvlab.postech.ac.kr/research/deconvnet/data/'

            for filename in filenames:
                wget.download(url + filename, out=os.path.join('data', filename))

                tar = tarfile.open(os.path.join('data', filename))
                tar.extractall(path='data')
                tar.close()

                os.remove(os.path.join('data', filename))


    def restore_session():
        global_step = 0
        if not os.path.exists(self.checkpoint_dir):
            raise IOError(self.checkpoint_dir + ' does not exist.')
        else:
            path = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if path is None:
                raise IOError('No checkpoint to restore in ' + self.checkpoint_dir)
            else:
                self.saver.restore(self.session, path.model_checkpoint_path)
                global_step = int(path.model_checkpoint_path.split('-')[-1])

        return global_step

    def predict(self, image):
        restore_session()
        return self.prediction.eval(session=self.session, feed_dict={image: [image]})[0]


    def train(self, train_stage=1, training_steps=5, restore_session=False, learning_rate=1e-6):
        if restore_session:
            step_start = restore_session()
        else:
            step_start = 0

        if train_stage == 1:
            trainset = open('data/stage_1_train_imgset/train.txt').readlines()
        else:
            trainset = open('data/stage_2_train_imgset/train.txt').readlines()

        for i in range(step_start, step_start+training_steps):
            # pick random line from file
            random_line = random.choice(trainset)
            image_file = random_line.split(' ')[0]
            ground_truth_file = random_line.split(' ')[1]
            image = np.float32(cv2.imread('data' + image_file))
            ground_truth = cv2.imread('data' + ground_truth_file[:-1], cv2.IMREAD_GRAYSCALE)
            # norm to 21 classes [0-20] (see paper)
            ground_truth = (ground_truth / 255) * 20
            print('run train step: '+str(i))
            start = time.time()
            self.train_step.run(session=self.session, feed_dict={self.x: [image], self.y: [ground_truth], self.rate: learning_rate})

            if i % 10000 == 0:
                print('step {} finished in {:.2f} s with loss of {:.6f}'.format(
                    i, time.time() - start, self.loss.eval(session=self.session, feed_dict={self.x: [image], self.y: [ground_truth]})))
                self.saver.save(self.session, self.checkpoint_dir+'model', global_step=i)
                print('Model {} saved'.format(i))


    def buildGraph(self):
        with tf.device(self.device):
            self.x = tf.placeholder(tf.float32, shape=(1, None, None, 3))
            self.y = tf.placeholder(tf.int64, shape=(1, None, None, self.NUM_CATEGORIES))
            expected = tf.expand_dims(self.y, -1)
            self.rate = tf.placeholder(tf.float32, shape=[])

            conv_1_1 = self.conv_layer(self.x, [5, 5, 3, 128], 128, 'conv_1_1')
            conv_1_2 = self.conv_layer(conv_1_1, [5, 5, 128, 192], 192, 'conv_1_2')

            pool_1, pool_1_argmax = self.pool_layer(conv_1_2)

            conv_2_1 = self.conv_layer(pool_1, [5, 5, 192, 192], 192, 'conv_2_1')
            conv_2_2 = self.conv_layer(conv_2_1, [5, 5, 192, 192], 192, 'conv_2_2')

            pool_2, pool_2_argmax = self.pool_layer(conv_2_2)

            conv_3_1 = self.conv_layer(pool_2, [5, 5, 192, 192], 192, 'conv_3_1')
            conv_3_2 = self.conv_layer(conv_3_1, [5, 5, 192, 192], 192, 'conv_3_2')

            pool_3, pool_3_argmax = self.pool_layer(conv_3_2)
            
            conv_4_1 = self.conv_layer(pool_3, [5, 5, 192, 192], 192, 'conv_4_1')
            conv_4_2 = self.conv_layer(conv_4_1, [5, 5, 192, 192], 192, 'conv_4_2')

            #image_lev_deconv = self.deconv_layer(conv_4_2, [1, 1, 96, 192], 96, 'image_lev_deconv')
            flattened = tf.reshape(conv_4_2, [-1, tf.shape(conv_4_2)[0] * tf.shape(conv_4_2)[1] * tf.shape(conv_4_2)[1]])
            dense_1 = tf.layers.dense(conv_4_2, units=1024, activation=tf.nn.relu)
            # dropout = tf.nn.dropout(dense, 0.30)
            dense_2 = tf.layers.dense(dense_1, units=self.NUM_CATEGORIES, activation=tf.nn.softmax)

            unpool_1 = self.unpool_layer2x2(conv_4_2, pool_3_argmax, tf.shape(conv_3_2))

            deconv_1 = self.deconv_layer(unpool_1, [5, 5, 192, 192], 192, 'deconv_1')
            element_wise_1 = tf.add(deconv_1, conv_3_2)
            concat_1 = self.concat_layer_broadcast(element_wise_1, dense_2, self.NUM_CATEGORIES)

            deconv_2 = self.deconv_layer(concat_1, [5, 5, 192, 210], 192, 'deconv_2')
            unpool_2 = self.unpool_layer2x2(deconv_2, pool_2_argmax, tf.shape(conv_2_2))

            deconv_3 = self.deconv_layer(unpool_2, [3, 3, 192, 192], 192, 'deconv_3')
            element_wise_2 = tf.add(deconv_3, conv_2_2)
            concat_2 = self.concat_layer_broadcast(element_wise_2, dense_2, self.NUM_CATEGORIES)

            deconv_4 = self.deconv_layer(concat_2, [5, 5, 192, 210], 192, 'deconv_4')
            unpool_3 = self.unpool_layer2x2(deconv_4, pool_1_argmax, tf.shape(conv_1_2))

            deconv_5 = self.deconv_layer(unpool_3, [3, 3, 192, 192], 192, 'deconv_5')
            element_wise_3 = tf.add(deconv_5, conv_1_2)
            concat_3 = self.concat_layer_broadcast(element_wise_3, dense_2, self.NUM_CATEGORIES)

            deconv_6 = self.deconv_layer(concat_3, [5, 5, 192, 192], 192, 'deconv_6')
            image_conv = self.conv_layer(self.x, [5, 5, 3, 192], 192, 'image_conv')
            element_wise_4 = tf.add(deconv_5, image_conv)

            conv_5 = self.conv_layer(element_wise_4, [3, 3, 192, 256], 256, 'conv_5')

            pred_deconv = self.deconv_layer(conv_5, [1, 1, 18, 256], 18, 'pred_deconv')
            element_wise_final = add_layer_broadcast(pred_deconv, dense_2, self.NUM_CATEGORIES)

            final_conv = self.conv_layer(element_wise_final, [1, 1, 18, 18], 18, 'final_conv')

            # score_1 = self.deconv_layer(prev, hparams['output_shape'], hparams['output_shape'][-2], 'score_1')
            # logits = tf.reshape(score_1, (-1, 21))
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(expected, [-1]), logits=logits, name='x_entropy')
            # self.loss = tf.reduce_mean(cross_entropy, name='x_entropy_mean')

            # self.train_step = tf.train.AdamOptimizer(self.rate).minimize(self.loss)

            # self.prediction = tf.argmax(tf.reshape(tf.nn.softmax(logits), tf.shape(score_1)), dimension=3)
            # self.accuracy = tf.reduce_sum(tf.pow(self.prediction - expected, 2))

    def add_layer_broadcast(self, layer, broadcast_layer, broadcast_layer_size):
        height = tf.shape(layer)[0]
        width = tf.shape(layer)[1]
        first_dim = tf.reshape(tf.tile(broadcast_layer, [height]), [height, broadcast_layer_size])
        full_dim = tf.stack([first_dim for i in range(width)], axis=1)
        return tf.add(layer, full_dim)

    def concat_layer_broadcast(self, layer, broadcast_layer, broadcast_layer_size):
        height = tf.shape(layer)[0]
        width = tf.shape(layer)[1]
        first_dim = tf.reshape(tf.tile(broadcast_layer, [height]), [height, broadcast_layer_size])
        full_dim = tf.stack([first_dim for i in range(width)], axis=1)
        return tf.concat([layer, full_dim], 2)

    def conv_layer(self, x, w_shape, b_shape, name, padding='SAME'):
        W = tf.Variable(tf.truncated_normal(w_shape, stddev = 0.1))
        b = tf.Variable(tf.constant(.1, shape=[b_shape]))
        return tf.nn.relu(tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding=padding) + b)

    def pool_layer(self, x):
        '''
        see description of build method
        '''
        with tf.device(self.device):
            return tf.nn.max_pool_with_argmax(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    def unravel_argmax(self, argmax, shape):
        output_list = []
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
        return tf.stack(output_list)

    def deconv_layer(self, x, w_shape, b_shape, name, padding='SAME'):
        W = tf.Variable(tf.truncated_normal(w_shape , stddev = 0.1))
        b = tf.Variable(tf.constant(.1, shape=[b_shape]))

        x_shape = tf.shape(x)
        out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[1], w_shape[2]])
        return tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b


    def unpool_layer2x2(self, x, raveled_argmax, out_shape):
        argmax = self.unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
        output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

        height = tf.shape(output)[0]
        width = tf.shape(output)[1]
        channels = tf.shape(output)[2]

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

        t2 = tf.squeeze(argmax)
        t2 = tf.stack((t2[0], t2[1]), axis=0)
        t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

        t = tf.concat([t2, t1], 3)
        indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

        x1 = tf.squeeze(x)
        x1 = tf.reshape(x1, [-1, channels])
        x1 = tf.transpose(x1, perm=[1, 0])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)

    def unpool_layer2x2_batch(self, x, argmax):
        '''
        Args:
            x: 4D tensor of shape [batch_size x height x width x channels]
            argmax: A Tensor of type Targmax. 4-D. The flattened indices of the max
            values chosen for each output.
        Return:
            4D output tensor of shape [batch_size x 2*height x 2*width x channels]
        '''
        x_shape = tf.shape(x)
        out_shape = [x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]]

        batch_size = out_shape[0]
        height = out_shape[1]
        width = out_shape[2]
        channels = out_shape[3]

        argmax_shape = tf.to_int64([batch_size, height, width, channels])
        argmax = unravel_argmax(argmax, argmax_shape)

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [batch_size*(width//2)*(height//2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, batch_size, height//2, width//2, 1])
        t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

        t2 = tf.to_int64(tf.range(batch_size))
        t2 = tf.tile(t2, [channels*(width//2)*(height//2)])
        t2 = tf.reshape(t2, [-1, batch_size])
        t2 = tf.transpose(t2, perm=[1, 0])
        t2 = tf.reshape(t2, [batch_size, channels, height//2, width//2, 1])

        t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

        t = tf.concat([t2, t3, t1], 4)
        indices = tf.reshape(t, [(height//2)*(width//2)*channels*batch_size, 4])

        x1 = tf.transpose(x, perm=[0, 3, 1, 2])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(out_shape))
        return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))

