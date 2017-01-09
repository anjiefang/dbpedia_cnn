# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

import os
import time
import datetime
import new_data_helpers as data_helpers
# from text_cnn_model import TextCNN_TFIDF as TextCNN
from NN_model import TextNN_TFIDF2 as TextCNN
from sklearn import metrics, cross_validation, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import csv
import sys
import json

# Parameters
# ==================================================
np.set_printoptions(threshold=sys.maxint)
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

# Data loading params
# tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")

# TODO: Philippine dataset
tf.flags.DEFINE_string("train_data_path", "None", "path to training dataset")
tf.flags.DEFINE_string("test_data_path", "None", "path to test dataset")
tf.flags.DEFINE_integer("n_class", "None", "number of classes")

# Model Hyperparameters
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("feature_num", 10000, "Feature number (default: 10000)")
0
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")



English=True

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


current_milli_time = lambda: int(round(time.time() * 1000))

# Data Preparatopn
# ==================================================

if FLAGS.train_data_path == "None":
    print "Please provide path to dataset..."
    sys.exit(1)

# Load data
print("Loading data...")

def load_data(path2data):
    data = []
    labels = []
    with open(path2data) as f:
        for line in f:
            line = line.split('\t')
            words = line[2]
            label = line[0]
            data.append(words)
            labels.append(label)
    return np.array(data), np.array(labels)


x_train, y_train = load_data(FLAGS.train_data_path)
x_test, y_test = load_data(FLAGS.test_data_path)

print x_train[:5]
print x_test[:5]
print y_train[:5]
print y_test[:5]
# raise

# Randomly shuffle data
np.random.seed(10)

shuffle_indices_train = np.random.permutation(np.arange(len(y_train)))
x_train = x_train[shuffle_indices_train]
y_train = y_train[shuffle_indices_train]

# shuffle_indices_test = np.random.permutation(np.arange(len(y_test)))
# x_test = x_test[shuffle_indices_test]
# y_test = y_test[shuffle_indices_test]

print x_train[:5]
print x_test[:5]
print y_train[:5]
print y_test[:5]


print "build TFIDF features..."
# vectorizer = TfidfVectorizer(max_features=500)
vectorizer = CountVectorizer(max_features=FLAGS.feature_num)
x_train = vectorizer.fit_transform(x_train).toarray()
x_test = vectorizer.transform(x_test).toarray()


print "map labels..."
print x_train[:5]
print x_test.shape
# raise

def map_label(n_class, labels):
    label_map = {}
    one_hot_labels = []
    for i in xrange(n_class):
        label_map[str(i)] = [1 if i==j else 0 for j in xrange(n_class)]
    for l in labels:
        one_hot_labels.append(label_map[l])
    return np.array(one_hot_labels)

y_train = map_label(FLAGS.n_class, y_train)
y_test = map_label(FLAGS.n_class, y_test)

print y_train[:5]
print y_test[:5]
# raise

# # Split train/test set
# y_labels = np.argmax(y_shuffled, axis=1)
# x_train, x_dev, y_train, y_dev = cross_validation.train_test_split(x_shuffled, y_shuffled, test_size=FLAGS.dev_sample_percentage, random_state=115, stratify=y_labels)
# print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
# print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
# # raise

# ''' cross validation '''
# y_labels = np.argmax(y_shuffled, axis=1)
# n_folds_data = cross_validation.StratifiedKFold(y_labels, n_folds=5)
#
# for train_index, dev_index in n_folds_data:
#     x_train, x_dev = x_shuffled[train_index], x_shuffled[dev_index]
#     y_train, y_dev = y_shuffled[train_index], y_shuffled[dev_index]
#     print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
#     print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            feature_size=x_train.shape[1],
            num_classes=FLAGS.n_class,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step_weighted(x_batch, y_batch):
            """
            A single training step
            """
            # r = 0.7
            # labels = np.argmax(y_batch, axis=1)
            # ratio_0 = 1 - len(labels[labels==0])/float(len(labels))
            # ratio_1 = 1 - len(labels[labels==1])/float(len(labels))
            # ratio_2 = 1 - len(labels[labels==2])/float(len(labels))

            # weighted_ratio = np.array([ratio_0, r*ratio_1, r*ratio_2]).reshape([1,3])
            # print weighted_ratio

            x_batch = np.array(x_batch).reshape([len(x_batch), 1, -1])
            print x_batch.shape

            y_batch = np.array(y_batch)

            labels = np.argmax(y_batch, axis=1)
            ratios = []
            for i in xrange(FLAGS.n_class):
                r = 1.0 - len(labels[labels==i])/float(len(labels))
                ratios.append(r)

            weighted_ratio = np.array(ratios).reshape(1, FLAGS.n_class)
            print weighted_ratio.shape
            # raise

            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
              cnn.weighted_ratio: weighted_ratio
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def train_step(x_batch, y_batch):
            """
            A single training step
            """

            x_batch = np.array(x_batch).reshape([len(x_batch), 1, -1])
            print x_batch.shape

            y_batch = np.array(y_batch)

            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            x_batch = np.array(x_batch).reshape([len(x_batch), 1, -1])
            y_batch = np.array(y_batch)

            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0,
              # cnn.weighted_ratio: np.ones([1,FLAGS.n_class])
            }
            step, summaries, loss, accuracy, predictions = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()

            ground = np.argmax(y_batch, axis=1)
            precision = metrics.precision_score(ground, predictions,labels=[i for i in xrange(FLAGS.n_class)], average=None)
            recall = metrics.recall_score(ground, predictions,labels=[i for i in xrange(FLAGS.n_class)], average=None)
            f1_score = metrics.f1_score(ground, predictions,labels=[i for i in xrange(FLAGS.n_class)], average=None)
            confusion = metrics.confusion_matrix(ground, predictions, labels=[i for i in xrange(FLAGS.n_class)])

            print("{}: step {}, loss {:g}, acc {:g}, precision: {}, recall: {}, f1: {}".format(time_str, step, loss, accuracy, precision, recall, f1_score))
            if writer:
                writer.add_summary(summaries, step)
            return precision, recall, f1_score, accuracy, confusion

        def predict_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0,
                cnn.weighted_ratio: np.ones([1,FLAGS.n_class])
            }
            step, summaries, loss, accuracy, predictions = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("1 bath prediction finished!")
            return predictions


        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)

            # weighted batch
            # train_step_weighted(x_batch, y_batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            # current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_test, y_test, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

        precision, recall, f1_score, accuracy, confusion, = dev_step(x_test, y_test, writer=dev_summary_writer)

        results = {}
        results['precision'] = precision.tolist()
        results['recall'] = recall.tolist()
        results['f1'] = f1_score.tolist()
        results['confusion'] = confusion.tolist()
        results['filters'] = FLAGS.filter_sizes
        results['accuracy'] = str(accuracy)

        savePath = FLAGS.train_data_path + '.' + str(current_milli_time()) + '.FN.' + str(FLAGS.feature_num) +'.res.json'
        with open(savePath, 'a') as f:
            dumped_result = json.dumps(results)
            f.write(dumped_result + '\n')

