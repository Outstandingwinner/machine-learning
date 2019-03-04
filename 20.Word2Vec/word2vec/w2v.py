#encoding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
filename = 'text8.zip'

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)


# Step 2: Build the dictionary and replace rare words with UNK token.
#vocabulary_size = len(words)
vocabulary_size = len(set(words))
print('Data size', vocabulary_size)
def build_dataset(words):
    count = [['UNK', -1]]
    #collections.Counter(words).most_common
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    data=[dictionary[word]  if  word in dictionary else 0 for word in words]
    #for word in words:
        #if word in dictionary:
            #index = dictionary[word]
        #else:
            #index = 0  # dictionary['UNK']
            #unk_count += 1
        #data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
# Sample data [14880, 4491, 483, 70, 1, 1009, 1850, 317, 14, 76]
#             ['欧几里得', '西元前', '三', '世纪', '的', '希腊', '数学家', '现在', '被', '认为']
# buffer= deque([4491, 483, 70, 1, 1009], maxlen=5)
# target = 483
# labels= [[   70]
#  [ 4491]
#  [    1]
#  [14880]
# buffer 和targrt同时右移一位
# target = 70
# labels= [[   70]
#  [ 4491]
#  [    1]
#  [14880]
#  [ 1009]
#  [    1]
#  [ 4491]
#  [  483]]


# 483 三 -> 70 世纪
# 483 三 -> 4491 西元前
# 483 三 -> 1 的
# 483 三 -> 14880 欧几里得
# 70 世纪 -> 1009 希腊
# 70 世纪 -> 1 的
# 70 世纪 -> 4491 西元前
# 70 世纪 -> 483 三

def generate_batch(batch_size, num_skips, skip_window):
    print("start-------------------")
    print("batch_size = %d, num_skips=%d, skip_window=%d"%(batch_size, num_skips, skip_window))
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):                  # 选5 (span)个词作为一组 输入buffer
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        print("buffer=",buffer)
        print("data_index=",data_index)
    for i in range(batch_size // num_skips):    #i取值0,1
        target = skip_window  # target label at the center of the buffer
        print("target=", target)
        targets_to_avoid = [skip_window]
        print("targets_to_avoid=", targets_to_avoid)
        for j in range(num_skips):  # 从buffer中选择4 (num_skips)个背景词  不等于target
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)  # target更新
            print("target=", target)
            targets_to_avoid.append(target)
            print("targets_to_avoid=",targets_to_avoid)
            batch[i * num_skips + j] = buffer[skip_window]
            print("batch[%d]=%d" %(i * num_skips + j,buffer[skip_window]))
            print("batch=", batch)
            labels[i * num_skips + j, 0] = buffer[target]
            print("labels[%d]=%d" % (i * num_skips + j, buffer[target]))
            print("labels=", labels)

        buffer.append(data[data_index])   # buffer 更新   buffer右移一位 target也右移一位
        print("buffer=", buffer)
        data_index = (data_index + 1) % len(data)

    print("batch=", batch)
    print("labels=", labels)
    print("end-------------------")
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=4, skip_window=2)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

# hyperparameters
batch_size = 128
embedding_size = 300 # dimension of the embedding vector
skip_window = 2 # how many words to consider to left and right
num_skips = 4 # how many times to reuse an input to generate a label

# we choose random validation dataset to sample nearest neighbors
# here, we limit the validation samples to the words that have a low
# numeric ID, which are also the most frequently occurring words
valid_size = 16 # size of random set of words to evaluate similarity on
valid_window = 100 # only pick development samples from the first 'valid_window' words
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64 # number of negative examples to sample

# create computation graph
graph = tf.Graph()

with graph.as_default():
    # input data
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    # operations and variables
    # look up embeddings for inputs
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                     labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))
    
    # construct the SGD optimizer using a learning rate of 1.0
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # compute the cosine similarity between minibatch examples and all embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # add variable initializer
    init = tf.initialize_all_variables()
#5
num_steps = 10

with tf.Session(graph=graph) as session:
    # we must initialize all variables before using them
    init.run()
    print('initialized.')
    
    # loop through all training steps and keep track of loss
    average_loss = 0
  
    for step in xrange(num_steps):
        # generate a minibatch of training data
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        
        # we perform a single update step by evaluating the optimizer operation (including it
        # in the list of returned values of session.run())
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        final_embeddings = normalized_embeddings.eval()
        print(final_embeddings)        
        print("*"*20)
        # print average loss every 2,000 steps
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # the average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0
        
        # computing cosine similarity (expensive!)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                # get a single validation sample
                valid_word = reverse_dictionary[valid_examples[i]]
                # number of nearest neighbors
                top_k = 8
                # computing nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "nearest to %s:" % valid_word
                for k in range(top_k):
                    print("=======================")
                    print("nearest=",nearest)
                    print("reverse_dictionary=",reverse_dictionary)
                    close_word = reverse_dictionary.get(nearest[k],None)
                    #close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
        
    final_embeddings = normalized_embeddings.eval()
    print(final_embeddings)
    fp=open('vector.txt','w',encoding='utf8')
    for k,v in reverse_dictionary.items():
        t=tuple(final_embeddings[k])
        #t1=[str(i) for i in t]
        s=''
        for i in t:
            i=str(i)
            s+=i+" "
            
        fp.write(v+" "+s+"\n")

    fp.close()
## Step 6: Visualize the embeddings.


#def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    #assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    #plt.figure(figsize=(18, 18))  # in inches
    #for i, label in enumerate(labels):
        #x, y = low_dim_embs[i, :]
        #plt.scatter(x, y)
        #plt.annotate(label,
                 #xy=(x, y),
                 #xytext=(5, 2),
                 #textcoords='offset points',
                 #ha='right',
                 #va='bottom')

    #plt.savefig(filename)

#try:
    #from sklearn.manifold import TSNE
    #import matplotlib.pyplot as plt

    #tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    #plot_only = 500
    #low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    #labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    #plot_with_labels(low_dim_embs, labels)

#except ImportError:
    #print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
