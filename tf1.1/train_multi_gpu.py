#! /usr/bin/env python
#coding:utf-8

import sys
import ConfigParser
import copy
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data
import math
from tensorflow.contrib import learn
import text_cnn


#parameters
tf.flags.DEFINE_integer("label_num",9,"the number of catalogs")
tf.flags.DEFINE_string("conf_path","./train_cnn.conf","")
#calculate 
def AverageGradients(tower_grads):
    with tf.name_scope('average_gradients'):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                if g is not None:
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)
            grad = tf.concat(grads,0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads

# Calculate loss and accuracy
def BatchLoss(text_length,input_x, input_y, word_index_dict_size,
        dropout_keep_prob, embedding_dim, filter_sizes, num_filters,
        x_size, l2_reg_lambda, cpus):
    """Calculate the total loss on a single tower running the CNN model.
    Args:
    scope: unique prefix string identifying the cnn tower, e.g. 'cnn_0'
    Returns:
    Tensor of shape [] containing the total loss for a batch of data
    """
    loss,accuracy = text_cnn.inference(
                                       input_x = input_x,
                                       input_y = input_y,
                                       sequence_length=text_length,
                                       vocab_size=word_index_dict_size,
                                       embedding_size=embedding_dim,
                                       filter_sizes=list(map(int, filter_sizes.split(","))),
                                       num_filters=num_filters,
                                       x_size=x_size,
                                       cpus=cpus,
                                       l2_reg_lambda=l2_reg_lambda,
                                       dropout_keep_prob = dropout_keep_prob)
    return loss, accuracy

def MultiGPUCalcLossAccu(text_length,gpus, input_x, input_y, batch_size, \
        word_index_dict_size, dropout_keep_prob, embedding_dim, \
        filter_sizes, num_filters, l2_reg_lambda, opt, global_step, cpus):
    tower_grads = []
    total_loss = []
    total_accuracy = []
    print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(len(gpus)):
            with tf.device('/gpu:%d' % gpus[i]):
                with tf.name_scope('%s_%d' % (text_cnn.TOWER_NAME, i)) as scope:

                    # Splice the inputs for each device
                    sx = tf.gather(input_x, range(i*batch_size/len(gpus), (i+1)*batch_size/len(gpus)))
                    sy = tf.gather(input_y, range(i*batch_size/len(gpus), (i+1)*batch_size/len(gpus)))
            
                    # Calculate loss and accuracy
                    loss, accuracy = BatchLoss(text_length, sx, sy,
                        word_index_dict_size, dropout_keep_prob, embedding_dim,
                        filter_sizes, num_filters, None, l2_reg_lambda, cpus)
                    tf.get_variable_scope().reuse_variables()
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
                    total_loss.append(loss)
                    total_accuracy.append(accuracy)
    with tf.device('/gpu:%d' % gpus[0]):
        grads = AverageGradients(tower_grads)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_op, variables_averages_op)
        average_loss = sum(total_loss)/len(total_loss)
        average_accuracy = sum(total_accuracy)/len(total_accuracy)
    return train_op, average_loss, average_accuracy

def TrainStep(sess, train_feed_dict, train_op, global_step, train_summary_op, average_loss,
        average_accuracy, train_summary_writer):
    _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, average_loss,
            average_accuracy], train_feed_dict)
    time_str = datetime.datetime.now().isoformat()
    train_summary_writer.add_summary(summaries, step)
    #if step % 100 == 0:
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

def DevLoss(text_length,dev_x, dev_y, word_index_dict_size, dropout_keep_prob,
        embedding_dim, filter_sizes, num_filters, l2_reg_lambda, cpus):
    x, y = data.LoadDevData(dev_path)
    x_size = len(y)
    tf.get_variable_scope().reuse_variables()
    dev_loss, dev_acc = BatchLoss(text_length, dev_x, dev_y,
             word_index_dict_size, dropout_keep_prob, embedding_dim,
             filter_sizes, num_filters, x_size, l2_reg_lambda, cpus)
    return dev_loss, dev_acc

def DevStep(sess, dev_feed_dict, dev_summary_op, dev_loss, dev_acc,
            current_step, writer=None):
    summaries, loss, accuracy = sess.run([dev_summary_op, dev_loss, dev_acc],
                                          dev_feed_dict)
    time_str = datetime.datetime.now().isoformat()
    if writer:
        writer.add_summary(summaries, current_step)
    print("{}, loss {:g}, acc {:g}".format(time_str, loss, accuracy))
    return accuracy

def SavePB(session, graph_name, model_dir, pb_name, current_step=''):
    var = {}
    for v in tf.trainable_variables():
        var[v.value().name] = session.run(v)
    g = tf.Graph()
    consts = {}
    with g.as_default(), tf.Session() as sess:
        for k in var.keys():
            consts[k] = tf.constant(var[k])
        tf.import_graph_def(graph_name.as_graph_def(), input_map={name:consts[name] for name in consts.keys()})
        tf.train.write_graph(sess.graph_def, model_dir, '%s.pb%s' % (pb_name, current_step), False)
        tf.train.write_graph(sess.graph_def, model_dir, '%s.txt%s' % (pb_name, current_step))

def RemovePBXStepsBefore(model_dir, pb_name, x_steps, current_step, checkpoint_every):
    suffix = str(current_step - x_steps * checkpoint_every)
    rm_file_txt = os.path.join( os.path.join(model_dir, 'models'), \
            pb_name + '.txt' + suffix )
    rm_file_pb = os.path.join( os.path.join(model_dir, 'models'), \
            pb_name + '.pb' + suffix )
    if os.path.exists(rm_file_txt):
         os.remove(rm_file_txt)
    if os.path.exists(rm_file_pb):
        os.remove(rm_file_pb)

def Train(train_path, dev_path, \
        word_index_dict_size, text_length, embedding_dim, \
        filter_sizes, num_filters, l2_reg_lambda, batch_size, \
        log_device_placement, model_dir, dropout_keep_prob_value, \
        num_epochs, evaluate_every, checkpoint_every, gpus, cpus):
    g1 = tf.Graph()
    valid_max_accuracy=0
    with g1.as_default(),tf.device("/gpu:" + str(gpus[0])):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement)
        config.gpu_options.allow_growth = True
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.Session(config=config)
        global_step = tf.get_variable('global_step', [], \
                initializer=tf.constant_initializer(0), trainable=False)
        opt = tf.train.AdamOptimizer(1e-3)
        #opt=tf.train.GradientDescentOptimizer(1e-3)



        input_x = tf.placeholder(tf.int32, [None, text_length], name="input_x")
        input_y = tf.placeholder(tf.float32, [None, 9], name="input_y")

        # For online predicting.
        dev_x = tf.placeholder(tf.int32, [None, text_length], name="dev_x")
        dev_y = tf.placeholder(tf.float32, [None, 9], name="dev_y")

        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        train_op, average_loss, average_accuracy = MultiGPUCalcLossAccu(
                text_length,gpus, input_x, input_y, batch_size,
                word_index_dict_size, dropout_keep_prob, embedding_dim,
                filter_sizes, num_filters, l2_reg_lambda, opt, global_step,
                cpus)
        dev_loss, dev_acc = DevLoss(text_length,dev_x, dev_y,
                word_index_dict_size, dropout_keep_prob, embedding_dim,
                filter_sizes, num_filters, l2_reg_lambda, cpus)

        # Session
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=2)
        ckpt=tf.train.get_checkpoint_state(model_dir)
        if reload_model and ckpt and ckpt.model_checkpoint_path:
            #_saver=tf.train.Saver(tf.global_variables()
            saver.restore(sess,ckpt.model_checkpoint_path)
            print "++++++++++++++++++++++****************"
            print sess.run(global_step)
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
	#config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement)
	#config.gpu_options.allow_growth = True
	#config.gpu_options.allocator_type = 'BFC'
	#sess = tf.Session(config=config)

	'''
        sess = tf.Session(config=tf.ConfigProto(
                          allow_soft_placement=True,
                          log_device_placement=log_device_placement))
	'''

        #sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        # Output directory for models and summaries
        out_dir = os.path.abspath(os.path.join(os.path.curdir, model_dir))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary =tf.summary.scalar("loss", average_loss)
        acc_summary =tf.summary.scalar ("accuracy", average_accuracy)
        dev_loss_summary = tf.summary.scalar("loss", dev_loss)
        dev_acc_summary =tf.summary.scalar ("accuracy", dev_acc)

        # Train Summaries
        print "Train Summaries"
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        print "Dev summaries"
        dev_summary_op = tf.summary.merge([dev_loss_summary, dev_acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
       
        for epoch in range(num_epochs):
            train_iter = data.BatchIter(train_path, batch_size)
            for train_batch in train_iter:
                x_batch, y_batch = zip(*train_batch)
                train_feed_dict = {input_x: x_batch,
                                   input_y: y_batch,
                                   dropout_keep_prob: dropout_keep_prob_value}
                """
                TrainBatch(train_feed_dict, sess, train_op, global_step, \
                        train_summary_op, average_loss, average_accuracy, \
                        train_summary_writer)
                """
                TrainStep(sess, train_feed_dict, train_op, global_step,
                        train_summary_op, average_loss, average_accuracy,
                        train_summary_writer)
         
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    x, y = data.LoadDevData(dev_path)
                    dev_feed_dict = {dev_x: x,
                                     dev_y: y,
                                     dropout_keep_prob:1.0}
		    #print("1111")
                    valid_accuracy=DevStep(sess, dev_feed_dict, dev_summary_op, dev_loss,
                            dev_acc, current_step, dev_summary_writer)
                    print("")

                current_step = tf.train.global_step(sess, global_step)
                if current_step % checkpoint_every == 0:
                    if valid_accuracy>valid_max_accuracy:
                        valid_max_accuracy=valid_accuracy
                        checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=current_step)
                    # Remove the old checkpoint files.
                    #RemovePBXStepsBefore(model_dir, 'cnn', 4, current_step, checkpoint_every)
                    # Save the latest checkpoint.
                    """
                    #save old meta data.
                    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=current_step)
                    """
                    #SavePB(sess, g1, os.path.join(model_dir, 'models'), 'cnn', \
                    #        current_step=str(current_step))

        #SavePB(sess, g1, os.path.join(model_dir, 'models'), 'cnn')
        sess.close()


if __name__ == '__main__':
    
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print FLAGS.label_num
    #if len(sys.argv) != 2:
    #    print "Invalid Format. Usage: python " + sys.argv[0] + " config_file"
    conf_file = FLAGS.conf_path
    conf = ConfigParser.ConfigParser()
    conf.read(conf_file)

    # Loading config params...
    # hyper_params
    embedding_dim = int(conf.get("hyper_param", "embedding_dim"))
    filter_sizes = conf.get("hyper_param", "filter_sizes")
    num_filters = int(conf.get("hyper_param", "num_filters"))
    dropout_keep_prob_value = float(conf.get("hyper_param", "dropout_keep_prob_value"))
    l2_reg_lambda = float(conf.get("hyper_param", "l2_reg_lambda"))
    # training_params
    batch_size = int(conf.get("training_param", "batch_size"))
    num_epochs = int(conf.get("training_param", "num_epochs"))
    evaluate_every = int(conf.get("training_param", "evaluate_every"))
    checkpoint_every = int(conf.get("training_param", "checkpoint_every"))
    reload_model=int(conf.get("training_param","reload_model"))
    #eval_size = int(conf.get("training_param", "eval_size"))
    # data_paths
    train_path = conf.get("data_path", "train_data")
    dev_path = conf.get("data_path", "dev_data")
    model_dir = conf.get("data_path", "model_dir")
    text_length = int(conf.get("data_path", "text_length"))
    vocab_file = conf.get("data_path", "vocab_file")
    # misc_params
    allow_soft_placement = int(conf.get("misc_param", "allow_soft_placement"))
    log_device_placement = int(conf.get("misc_param", "log_device_placement"))
    # devices
    gpus_str = conf.get("device", "gpus")
    cpus_str = conf.get("device", "cpus")
    gpus = [int(i) for i in gpus_str.strip().split(',')]
    cpus = [int(i) for i in cpus_str.strip().split(',')]

    """
    # Loading Training Data...
    load_start_time = time.time()
    x_train1, x_train2, x_train3, x_dev1, x_dev2, x_dev3, word_index_dict_size \
            = data.LoadTrainData(train_path, vocab_file, eval_size)
    load_end_time = time.time()
    print "load_data_time_consumption: %s second(s)\n" % (load_end_time - load_start_time)
    """
    load_start_time = time.time()
    word_index_dict_size = len(data.load_word_index_dict(vocab_file))
    print "word_index_dict_size:%d\n"%(word_index_dict_size)
    load_end_time = time.time()
    print "load word_index time consumption: %s second(s)\n" % (load_end_time - load_start_time)


    # Training... (including checkpoints model)
    train_start_time = time.time()
    Train(train_path, dev_path,
            word_index_dict_size, text_length, embedding_dim,
            filter_sizes, num_filters, l2_reg_lambda, batch_size,
            log_device_placement, model_dir, dropout_keep_prob_value,
            num_epochs, evaluate_every, checkpoint_every, gpus, cpus)
    train_end_time = time.time()
    print "training_time_consumption: %s hour(s)\n" % ((train_end_time - train_start_time) / 3600)

    # Writing the Final Models(pb format)
    output_start_time = time.time()
    #SavePB(sess, g1, os.path.join(model_dir, 'models'), 'cnn')
    output_end_time = time.time()
    output_time_consumption = (output_end_time - output_start_time) / 3600

    # Logging.
    with open(os.path.join(model_dir, 'log'), 'w') as f:
        f.write('num_epochs:%s\n' % num_epochs)
        f.write('sequence_length:%s\n' % text_length)
        f.write('output_start_time:%s\n' % output_start_time)
        f.write('output_end_time:%s\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
        f.write('output_time_consumption:%s hour(s)\n' % output_time_consumption)
