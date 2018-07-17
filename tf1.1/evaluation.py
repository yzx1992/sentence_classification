import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data
from tensorflow.contrib import learn
import sys


def get_max_index(lst):
    if len(lst)<=0:
       return (-1,-1)
    max2=lst[0]
    max_index=0
    for i in range(1,len(lst)):
       if lst[i]>max2:
          max2=lst[i]
          max_index=i
    return (max2,max_index)

def get_top_n_indexs_values(lst,n):
        _len=len(lst)
	if _len<n:
            n=_len
	if _len<=0:
		return [(-1,-1) for i in range(n)]
	
	
	_x=dict()
	for i in range(_len):
		_x[i]=lst[i]
	
	_x=sorted(_x.iteritems(),key=lambda _x:_x[1],reverse=True)
 #       print _x[0:n]	
	return _x[0:n]

	
def get_index(ary):
    index=-1
    for i in range(len(ary)):
       if ary[i]==1:
          index=i
          break
    return index



cat_dict={2:"caiwu",3:"ditu",4:"gouwu",5:"K12",6:"meishi",7:"yiliao",8:"yuedu",9:"yule",1:"other"}




tf.flags.DEFINE_string("test_data", "/mnt/workspace/yezhenxu/data/charlevel/len100/test/topic.s2id", "")
tf.flags.DEFINE_string("model_dir", "/mnt/workspace/yezhenxu/model/char_model/tf1.0-len100", "")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

x, y = data.LoadDevData(FLAGS.test_data)

cat_num_ary=[0,0,0,0,0,0,0,0,0]



for i in range(len(y)):
    index=get_index(y[i])
    if index<0:
        continue
    cat_num_ary[index]=cat_num_ary[index]+1
   
total_sample_num=len(y)
print "total_sample_num= %d"%(total_sample_num)






def top1():
    cat_predict_right_num_ary = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_allpredict_ary = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    all_predict_right_num = 0

    index = 0
    for seq in batch_probs:
        max_val, max_index = get_max_index(seq)
        if max_index < 0:
            index = index + 1
            continue

        if y[index][max_index] == 1:
            all_predict_right_num = all_predict_right_num + 1
            cat_predict_right_num_ary[max_index] = cat_predict_right_num_ary[max_index] + 1

        cat_allpredict_ary[max_index] = cat_allpredict_ary[max_index] + 1
        index = index + 1

    print "top1 all:"
    all_accuracy_rate = float(float(all_predict_right_num) / total_sample_num) * 100
    print "all_predict_right_num:%d, total_sample_num:%d,all_accuracy_rate:%f%s" %(all_predict_right_num, total_sample_num, all_accuracy_rate, "%")

    print "----------------------------------------"
    print "per catalog:"
    for i in range(len(cat_predict_right_num_ary)):
        recall_rate = float(float(cat_predict_right_num_ary[i]) / cat_num_ary[i]) * 100
        precision_rate = float(float(cat_predict_right_num_ary[i]) / cat_allpredict_ary[i]) * 100
        f1_rate = float((2 * (recall_rate * precision_rate)) / (recall_rate + precision_rate)) 
        print "%s catalog, predict_right_num:%d, total_num:%d,predict_all_num:%d precision_rate:%f%s,recall_rate:%f%s,f1_rate:%f%s"%(cat_dict[i + 1], cat_predict_right_num_ary[i], cat_num_ary[i], cat_allpredict_ary[i], precision_rate, "%",
            recall_rate, "%", f1_rate, "%")


def top(n):
    _cat_predict_right_num_ary = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    _cat_allpredict_ary = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    _all_predict_right_num = 0
    j = 0
    for seq in batch_probs:
    
        indexs_values = get_top_n_indexs_values(seq.tolist(), n)
    
        for i in range(len(indexs_values)):
            _index = indexs_values[i][0]

            if _index < 0:
                j += 1
                continue

            if y[j][_index] == 1:
                _all_predict_right_num = _all_predict_right_num + 1
                _cat_predict_right_num_ary[_index] = _cat_predict_right_num_ary[_index] + 1

        j = j + 1

    print "top%d all:" %(n)
    all_accuracy_rate = float(float(_all_predict_right_num) / total_sample_num) * 100
    print "all_predict_right_num:%d, total_sample_num:%d,all_accuracy_rate:%f%s"%(_all_predict_right_num,
                                                                                  total_sample_num, all_accuracy_rate,
                                                                                  "%")

    print "----------------------------------------"
    print "per catalog:"
    for i in range(len(_cat_predict_right_num_ary)):
        recall_rate = float(float(_cat_predict_right_num_ary[i]) / cat_num_ary[i]) * 100
        print "%s catalog, predict_right_num:%d, total_num:%d,recall_rate:%f%s"%(cat_dict[i + 1],
                                                                                _cat_predict_right_num_ary[i],
                                                                                cat_num_ary[i], recall_rate, "%")


graph = tf.Graph()
#saver=tf.train.Saver(tf.global_variables())

with graph.as_default(), tf.device('/gpu:6'):
   # output_graph_def = tf.GraphDef()
   # output_graph_path = FLAGS.model_dir
   # with open(output_graph_path, 'rb') as f:
    #    output_graph_def.ParseFromString(f.read())
     #   _ = tf.import_graph_def(output_graph_def, name='')
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
   
    saver=tf.train.import_meta_graph(FLAGS.model_dir+'/model.ckpt-2500.meta')
    #ckpt=tf.train.get_checkpoint_state(moder_dir)
    #if ckpt and ckpt.model_checkpoint_path:
     #   saver.restore(sess,ckpt.model_checkpoint_path)
    #else:
     #   print "No checkpoint file found!"
        
    with sess.as_default():
    
       # tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()
        input_x = sess.graph.get_tensor_by_name('dev_x:0')
        input_y = sess.graph.get_tensor_by_name('dev_y:0')
        dropout_keep_prob = sess.graph.get_tensor_by_name('dropout_keep_prob:0')
        scores = sess.graph.get_tensor_by_name('output/scores:0') 
        probs = sess.graph.get_tensor_by_name('output/probs:0') 
        saver.restore(sess,tf.train.latest_checkpoint(FLAGS.model_dir))
        #saver.restore(sess,FLAGS.model_dir+'/model.ckpt-2500.data-00000-of-00001')        
        batch_scores, batch_probs = sess.run([scores, probs], {input_x: x, input_y: y, dropout_keep_prob:1.0})
        #print len(batch_scores)
        #print len(batch_probs
        print ("*****************************\n")
        top1()    
        print ("*****************************\n")
        top(3)
