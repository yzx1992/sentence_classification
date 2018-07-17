import numpy as np
import re
import itertools
import operator
from collections import Counter


def input_data(train_data):
    with open(train_data, "r") as f:

        lines = list(f.readlines())
        lines = [s.strip() for s in lines]
        #print lines
        a = zip(*[s.split("\t") for s in lines])
        #print a
        #print a
        x = [s.strip() for s in a[1]]
        l = list(a[0])
        y = []
       # print l
        for i in l:
            #print i
            y.append([0]*9)
            #print y
            y[len(y)-1][int(i)-1] = 1
            #print y

    #print y
    return [x, y]

def BatchIter(data_path, batch_size):
    #print(data_path)
    with open(data_path, 'r') as f:
        sample_num = 0
        samples = []
        for line in f:
            line = line.strip()
            samples.append(line)
            sample_num += 1
            if sample_num == batch_size:
                a = zip(*[s.split("\t") for s in samples])
                l = list(a[0])
                x = [s.strip() for s in a[1]]
                x = np.array(text2list(x))
                y = []
                for i in l:
                    y.append([0]*9)
                    y[len(y)-1][int(i)-1] = 1
                batch_data = list(zip(x, y))
                batch_data = np.array(batch_data)
                yield batch_data
                sample_num = 0
                samples = []
        """
        if len(samples) > 0:
            a = zip(*[s.split("\t") for s in samples])
            x_text1 = [s.strip() for s in a[0]]
            x_text2 = [s.strip() for s in a[1]]
            x_text3 = [s.strip() for s in a[2]]
            x1 = np.array(text2list(x_text1))
            x2 = np.array(text2list(x_text2))
            x3 = np.array(text2list(x_text3))
            batch_data = list(zip(x1, x2, x3))
            batch_data = np.array(batch_data)
            yield batch_data
        """

def batch_iter(data, batch_size, num_epochs, shuffle=True, train=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if train:
                if (batch_num+1)*batch_size > data_size :
                    continue
            yield shuffled_data[start_index:end_index]

def load_word_index_dict(word_index_file):
    lines = list(open(word_index_file, "r").readlines())
    lines = [s.strip() for s in lines]
    word_index_dict = {}
    num=0
    for line in lines:
        s = line.split('\t')
        if len(s) != 2:
            num=num+1
	    print "error:%s"%(line)
            print num	
            continue
        word = s[0].strip()
        if word not in word_index_dict:
            #word_index_dict[word] = int(s[1].strip())-num
            
            word_index_dict[word] = int(s[1].strip())
    return word_index_dict

def text2list(word_id_text):
    list4np = []
    for text in word_id_text:
        text_secs = text.strip().split(' ')
        text_line = []
        for i in range(0, len(text_secs)):
            text_line.append(int(text_secs[i]))
        list4np.append(text_line)
    return list4np

#load train data
def LoadTrainData(train_data, word_index_file, eval_size):
    print("Loading data...")
    x_text1, x_text2, x_text3 = input_data(train_data)

    #generate word_id training-data(x1, x2, x3) from word_index_dict
    print "Loading word index file ..."
    word_index_dict = load_word_index_dict(word_index_file)

    # x1,x2,x3 are word_id tensors.
    x1 = np.array(text2list(x_text1))
    print "x1"
    print  x1
    x2 = np.array(text2list(x_text2))
    x3 = np.array(text2list(x_text3))

    # Split train/test set. This is very crude, should use cross-validation
    sampleSize = len(x1)
    print "eval_size:%d" % eval_size
    devSize = sampleSize / eval_size
    x_train1, x_dev1 = x1[:-devSize], x1[-devSize:]
    x_train2, x_dev2 = x2[:-devSize], x2[-devSize:]
    x_train3, x_dev3 = x3[:-devSize], x3[-devSize:]

    print("Vocabulary Size: {:d}".format(len(word_index_dict)))
    return x_train1, x_train2, x_train3, x_dev1, x_dev2, x_dev3, len(word_index_dict)

def LoadDevData(dev_path):
    x, y = input_data(dev_path)
    x = np.array(text2list(x))
    y = np.array(y)
    return x, y
