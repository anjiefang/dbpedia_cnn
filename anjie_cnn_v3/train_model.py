# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np

import os
import time
import datetime
import new_data_helpers as data_helpers
from text_cnn_model import TextCNN_Weighted as TextCNN
# from text_cnn_model import TextCNN
from sklearn import metrics, cross_validation, preprocessing
import csv
import sys
import re
from EV_utils import get_WE_vectors, initialise_UNK
import gzip

# Parameters
# ==================================================
np.set_printoptions(threshold=sys.maxint)
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")

# TODO: Philippine dataset
tf.flags.DEFINE_string("data_path", "None", "Data source")
tf.flags.DEFINE_string("we_vocab_path", "None", "Vocabulary file")
tf.flags.DEFINE_string("we_vector_path", "None", "Vector file")
tf.flags.DEFINE_integer("n_class", "None", "number of classes")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 500, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many steps (default: 100)")
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

# Data Preparatopn
# ==================================================

if FLAGS.data_path == "None":
    print "Please provide path to dataset..."
    sys.exit(1)
if FLAGS.we_vocab_path == "None":
    print "Please provide path to embedding vocabulary..."
    sys.exit(1)
if FLAGS.we_vector_path == "None":
    print "Please provide path to embedding vector..."
    sys.exit(1)

# Load data
print("Loading data...")
we_vocab = data_helpers.load_WE_vocab(FLAGS.we_vocab_path)
x_text, y = data_helpers.load_data_and_labels(FLAGS.data_path, we_vocab=we_vocab, n_class=FLAGS.n_class , english=English)


def tokenizer(iterator):
    for value in iterator:
        yield value.split(" ")

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, tokenizer_fn=tokenizer)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

dataset_map =  vocab_processor.vocabulary_._mapping

def build_we_indices(dataset_map, we_vocab):
    we_indices = []
    #sort the data vocab by index
    data_vocab = sorted(dataset_map.items(), key=lambda x: x[1])
    words, data_indices = zip(*data_vocab)
    idx = 0
    for w in words:
        if w =="<UNK>":
            if idx != 0:
                raise("UNK index not zero")
            continue
        #get word index in we vocab
        we_indices.append(we_vocab[w])
        idx += 1
    return we_indices

we_indices = build_we_indices(dataset_map, we_vocab)

# TODO: change the path
we_vectors = get_WE_vectors(vectorPath=FLAGS.we_vector_path,
                            we_length=len(we_vocab), indices=we_indices, npy=True)

print we_vectors.shape
we_vectors = initialise_UNK(we_vectors)
print we_vectors.shape
print x[1]

# print "loading unseen"
# for unseen_data in data_helpers.load_unseen_data(FLAGS.unseen_data_path):
#     tids, unseen_text = zip(*unseen_data)
#     transformed_text = np.array(list(vocab_processor.transform(unseen_text)))
#     y_prediction = np.array([[1, 0, 0] for i in xrange(len(transformed_text))])
#     print transformed_text[1]
#     print y_prediction[1]
#     print transformed_text.shape
#     print y_prediction.shape
#     raise


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
            sequence_length=x_train.shape[1],
            num_classes=FLAGS.n_class,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
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

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())
        ''' override WE in CNN setup'''
        sess.run(cnn.W.assign(we_vectors))

        def train_step_weighted(x_batch, y_batch):
            """
            A single training step
            """
            r = 0.7
            labels = np.argmax(y_batch, axis=1)
            ratio_0 = 1 - len(labels[labels==0])/float(len(labels))
            ratio_1 = 1 - len(labels[labels==1])/float(len(labels))
            ratio_2 = 1 - len(labels[labels==2])/float(len(labels))

            weighted_ratio = np.array([ratio_0, r*ratio_1, r*ratio_2]).reshape([1,3])
            # print weighted_ratio

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

            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0,
              cnn.weighted_ratio: np.ones([1,3])
            }
            step, summaries, loss, accuracy, predictions = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()

            ground = np.argmax(y_batch, axis=1)
            precision = metrics.precision_score(ground, predictions,labels=[0,1,2], average=None)
            recall = metrics.recall_score(ground, predictions,labels=[0,1,2], average=None)
            f1_score = metrics.f1_score(ground, predictions,labels=[0,1,2], average="macro")

            print("{}: step {}, loss {:g}, acc {:g}, precision: {}, recall: {}, f1: {:g}".format(time_str, step, loss, accuracy, precision, recall, f1_score))
            if writer:
                writer.add_summary(summaries, step)
            return f1_score

        def predict_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0,
                cnn.weighted_ratio: np.ones([1,3])
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
            train_step_weighted(x_batch, y_batch)

            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                f1 = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                #vz 0.68
                #ph 0.72
                # if f1 > 0.72:
                #     print("good results achieved!")
                #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                #     print("Saved model checkpoint to {}\n".format(path))
                #     break
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

