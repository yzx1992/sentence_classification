import tensorflow as tf
import numpy as np

TOWER_NAME = 'CNN'

def _variable_on_cpu(name, shape, initializer, cpus):
    with tf.device('/gpu:6' ):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def inference(input_x, input_y, sequence_length,
        vocab_size, embedding_size, filter_sizes, num_filters,
        x_size, cpus, l2_reg_lambda=0.0, dropout_keep_prob=0.5):
    """
    Build the cnn model.
    Args:
    input_x:
    Returns:
    Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().

    # Keeping track of l2 regularization loss (optional)
    l2_loss = tf.constant(0.0)

    # Embedding layer
    with tf.variable_scope("embedding") as scope:
        W = _variable_on_cpu("W", [vocab_size+22, embedding_size],
                tf.random_uniform_initializer(-1.0, 1.0), cpus)
        print "ok"
        print vocab_size,input_x
        embedded_chars = tf.nn.embedding_lookup(W, input_x)
        # print embedded_chars
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        # print "-------"
        # print embedded_chars_expanded
    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("conv-maxpool-%s" % filter_size) as scope :
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = _variable_on_cpu("W", filter_shape, tf.truncated_normal_initializer(stddev=0.1), cpus)
            b = _variable_on_cpu('b', [num_filters],tf.constant_initializer(0.1), cpus)
            conv = tf.nn.conv2d(embedded_chars_expanded,
                               W,
                               strides=[1, 1, 1, 1],
                               padding="VALID",
                               name="conv")
	    print "----------"
	    print conv
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(h,
                                  ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                  strides=[1, 1, 1, 1],
                                  padding='VALID',
                                  name="pool")
	    print "-----------------"
	    print pooled
            pooled_outputs.append(pooled)
    # Combine all the pooled features
    with tf.variable_scope("combine") as scope :
        num_filters_total = num_filters * len(filter_sizes)
	print "combine"
	print pooled_outputs
	print "1111"
	print "---"
        h_pool = tf.concat( pooled_outputs,3)
	print h_pool
	print "222"
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total], name="encode")
	print h_pool_flat
    # Add dropout
    with tf.variable_scope("dropout") as scope :
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob, name="h_drop")
 
    # 4 Hidden layer to map all the pooled features
    with tf.variable_scope("output") as scope:
        #W = _variable_on_cpu("W1", [num_filters_total, 2], tf.truncated_normal_initializer(stddev=0.1), cpus)
        #b = _variable_on_cpu('b1', [2], tf.constant_initializer(0.1), cpus)

        W = _variable_on_cpu("W1", [num_filters_total, 9], tf.truncated_normal_initializer(stddev=0.1), cpus)
        b = _variable_on_cpu('b1', [9], tf.constant_initializer(0.1), cpus)
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
	print "h_drop"
	print h_drop.get_shape()
        scores = tf.nn.xw_plus_b(h_drop, W, b, name='scores')
	print "scores shape"
	print scores
        probs = tf.nn.softmax(scores, name='probs')
        predictions = tf.argmax(scores, 1, name = 'predictions')


    # CalculateMean cross-entropy loss
    with tf.variable_scope("loss") as scope :
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=input_y,logits=scores)
        loss = tf.reduce_mean(losses) + l2_reg_lambda*l2_loss

    # Accuracy
    with tf.name_scope("accuracy"):
        correct = tf.equal(predictions, tf.argmax(input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")
    return loss, accuracy
