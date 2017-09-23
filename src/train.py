import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import csv
import pickle
from tensorflow.python import debug as tf_debug

logging.basicConfig(level=logging.INFO)

DIR_DATASET = "../dataset"
FILENAME_PKL = "train.pkl"

PATH_IMG_PKL = os.path.join(DIR_DATASET, FILENAME_PKL)
PATH_LABELS = os.path.join(DIR_DATASET, "label.txt")
PATH_SEGM_LABELS = os.path.join(DIR_DATASET, "segm_labels.csv")

# Load dataset images and labels
labels = []
if os.path.exists(PATH_SEGM_LABELS):
    with open(PATH_SEGM_LABELS, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        for line in csv_reader:
            train_label = np.array(line, np.uint8)
            labels.append(train_label)
            logging.debug("label:{:}".format(train_label))
            logging.debug("label.shape:{:}".format(train_label.shape))

with open(PATH_IMG_PKL, "rb") as f:
    imgs = pickle.load(f)
    logging.info(len(imgs))

split_rate = 0.7
split = int(len(imgs) * split_rate)
train_datas, train_labels = imgs[:split], labels[:split]
test_datas, test_labels = imgs[split:], labels[split:]

# logging.info('train_data.shape:{:}'.format(train_datas.shape))
# logging.info('train_label.shape:{:}'.format(train_labels.shape))
# logging.info('test_data.shape:{:}'.format(test_datas.shape))
# logging.info('test_label.shape:{:}'.format(test_labels.shape))

# for i in range(5):
#     plt.imshow(np.squeeze(train_datas[i+5]), cmap='gray')
#     plt.title(train_labels[i+5])
#     plt.show()

n_classes = 11
epochs = 150
batch_size = 128
DIR_CKPT = "../ckpt"

x = tf.placeholder(tf.float32, [None, None, None, 1], name='x')
y = tf.placeholder(tf.uint8, [1], name="y")

def residual_sepconv_block(x, filters, name):
    with tf.variable_scope(name):
        relu1 = tf.nn.relu(x)
        sepconv1 = tf.layers.separable_conv2d(relu1, filters, [3, 3],
                                              strides=1,
                                              padding='same',
                                              name="sepconv1")
        relu2 = tf.nn.relu(sepconv1)
        sepconv2 = tf.layers.separable_conv2d(relu2, filters, [3, 3],
                                              strides=1,
                                              padding='same',
                                              name="sepconv2")
        identity_mapping = tf.layers.conv2d(x, filters, [1, 1],
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                            bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                            name="identity_mapping")
    return identity_mapping + sepconv2

with tf.variable_scope('conv'):
    conv1 = tf.layers.conv2d(inputs=x,
                             filters=32,
                             kernel_size=[3, 3],
                             strides=1,
                             padding='same',
                             activation=tf.nn.relu,
                             name='conv1')
    conv2 = tf.layers.conv2d(inputs=conv1,
                             filters=64,
                             kernel_size=[3, 3],
                             strides=1,
                             padding='same',
                             activation=tf.nn.relu,
                             name='conv2')
    # resconv1 = residual_sepconv_block(conv2, 8, "residual_1")
    # resconv2 = residual_sepconv_block(resconv1, 16, "residual_2")

with tf.variable_scope('fc'):
    # conv3 = tf.reshape(resconv2, [-1, int(np.prod(resconv2.shape[1:]))])
    average_pooling = tf.reduce_mean(conv2, axis=[1, 2])
    fc = tf.layers.dense(inputs=average_pooling,
                         units=32,
                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                         bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                         activation=tf.nn.relu)

with tf.variable_scope('logits'):
    # logit shape (batch, n_classes)
    logits = tf.layers.dense(fc, n_classes)
    logits = tf.identity(logits, name="logits")

with tf.name_scope('xent'):
    y_one_hot = tf.one_hot(y, n_classes)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot))

optimizer = tf.train.AdamOptimizer().minimize(cost)

pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
accuracy_char = tf.reduce_mean(tf.cast(pred, tf.float32), name='accuracy_char')

logging.info('model build!')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    init = tf.global_variables_initializer()
    sess.run(init)
    i, loss_accum, train_acc_accum = 0, 0, 0
    for epoch in range(epochs):
        for train_data, train_label in zip(train_datas, train_labels):
            i += 1
            sess.run(optimizer, feed_dict={x: train_data, y: train_label})
            loss_accum, train_acc = sess.run([cost, accuracy_char], feed_dict={x: train_data, y: train_label})
            loss_accum += loss_accum
            train_acc_accum += train_acc
        logging.info("epoch:{:} i:{:} training loss:{:<5} accuracy_char:{:<5}"
                     .format(epoch, i, loss_accum / i, train_acc_accum / i))

# Test
    i, loss_accum, test_acc_accum = 0, 0, 0
    for test_data, test_label in zip(test_datas, test_labels):
        i += 1
        sess.run(optimizer, feed_dict={x: test_data, y: test_label})
        loss_accum, test_acc = sess.run([cost, accuracy_char], feed_dict={x: test_data, y: test_label})
        loss_accum += loss_accum
        test_acc_accum += test_acc
    logging.info("i:{:} test loss:{:<5} test_accuracy_char:{:<5}"
                 .format(i, loss_accum / i, test_acc_accum / i))

    PATH_CKPT = os.path.join(DIR_CKPT, "model.ckpt")
    saver = tf.train.Saver()
    saver.save(sess, PATH_CKPT, global_step=epochs)
    logging.info("Model saved in:{:}".format(DIR_CKPT))

