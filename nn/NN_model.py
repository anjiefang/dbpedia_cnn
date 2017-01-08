import tensorflow as tf

class TextNN_TFIDF(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, feature_size, num_classes, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout

        self.input_x = tf.placeholder(tf.float32, [None, 1, feature_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        print feature_size, num_classes, filter_sizes, num_filters, l2_reg_lambda

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        # tf.set_random_seed(1234)
        reshaped_x = tf.reshape(self.input_x, [-1, feature_size])

        with tf.name_scope("layer1"):
            num_layer1 = 3000
            W = tf.Variable(tf.truncated_normal([feature_size, num_layer1], stddev=0.1), name="layer1_W")
            b = tf.Variable(tf.constant(0.1, shape=[num_layer1]), name="layer1_b")
            layer1_score =tf.nn.xw_plus_b(reshaped_x, W, b, name="layer1_scores")
            # Apply nonlinearity
            layer1_h = tf.nn.relu(tf.nn.bias_add(layer1_score, b), name="layer1_relu")

        # Add dropout
        with tf.name_scope("dropout"):
            self.layer1_drop = tf.nn.dropout(layer1_h, self.dropout_keep_prob)

        with tf.name_scope("layer2"):
            num_layer2 = 3000
            W = tf.get_variable(
                "layer2_W",
                shape=[num_layer1, num_layer2],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[num_layer2]), name="layer2_b")
            layer2_score = tf.nn.xw_plus_b(self.layer1_drop, W, b, name="layer2_scores")
            layer2_h = tf.nn.relu(tf.nn.bias_add(layer2_score, b), name="layer2_relu")
            layer2_drop = tf.nn.dropout(layer2_h, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            # W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            W = tf.get_variable(
                "W",
                shape=[num_layer2, num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(layer2_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")