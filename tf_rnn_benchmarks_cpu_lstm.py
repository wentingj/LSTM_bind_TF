#!/usr/bin/env python
import time
import optparse
import numpy as np
import tensorflow as tf
#from tensorflow.python.ops import rnn
from tensorflow.contrib import rnn
from tensorflow.python.ops import variables as tf_variables

def get_feed_dict(x_data, y_data=None):
    feed_dict = {}

    if y_data is not None:
        feed_dict[y] = y_data

    #for i in xrange(x_data.shape[0]):
    for i in range(x_data.shape[0]):
        feed_dict[x[i]] = x_data[i, :, :]

    return feed_dict


# Parameters
optparser = optparse.OptionParser()
optparser.add_option("-n", "--network_type", default='rnn', help="Network type (rnn, lstm, basic_lstm, blstm)")
optparser.add_option("-i", "--input_size", default=150, type='int', help="Input layer size")
optparser.add_option("-l", "--hidden_size", default=1024, type='int', help="Hidden layer size")
optparser.add_option("-s", "--seq_length", default=10, type='int', help="Sequence length")
optparser.add_option("-b", "--batch_size", default=64, type='int', help="Batch size")
opts = optparser.parse_args()[0]

network_type = opts.network_type
print(network_type)
hidden_size = opts.hidden_size
print("hidden_size=", hidden_size)
input_size = opts.input_size
print("input_size=", input_size)
seq_length = opts.seq_length
print("seq_length=", seq_length)
batch_size = opts.batch_size
print("batch_size=", batch_size)

n_batch = 1000
n_samples = batch_size * n_batch 
loops = 1
forward_samples = batch_size * loops

if network_type == 'lstm' or network_type == 'basic_lstm' or network_type == 'rnn' or network_type == 'gru':
    # Data
    xinput = np.random.rand(seq_length, batch_size, input_size).astype(np.float32)
    xinput_intel = np.transpose(xinput, (0, 2, 1))
    ytarget = np.random.rand(batch_size, hidden_size).astype(np.float32)
    x = [tf.placeholder(tf.float32, [batch_size, input_size], name="x") for i in range(seq_length)]
    y = tf.placeholder(tf.float32, [batch_size, hidden_size], name="y")
    init_state = [tf.zeros([batch_size, hidden_size]), tf.zeros([batch_size, hidden_size])]
elif network_type == 'blstm':
    xinput = np.random.rand(seq_length, batch_size, input_size).astype(np.float32)
    ytarget = np.random.rand(batch_size, 2*hidden_size).astype(np.float32)
    x = [tf.placeholder(tf.float32, [batch_size, input_size], name="x") for i in range(seq_length)]
    y = tf.placeholder(tf.float32, [batch_size, 2*hidden_size], name="y")
else:
    raise Exception('Unknown network! '+network_type)

if network_type == 'rnn':
    cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
elif network_type == 'lstm':
    cell = tf.nn.rnn_cell.LSTMCell(hidden_size, hidden_size)
elif network_type == 'basic_lstm':
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0)
elif network_type == 'gru':
    cell = tf.nn.rnn_cell.GRUCell(hidden_size)
elif network_type == 'blstm':
    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
else:
    raise Exception('Unknown network! '+network_type)

# Thread parallelism
inter_op = 1
intra_op = 66

print("Compiling...")
start = time.time()
if network_type == 'lstm' or network_type == 'basic_lstm' or network_type == 'rnn' or network_type == 'gru':
    output, _cell_state = rnn.static_rnn(cell, x, dtype=tf.float32, initial_state=init_state)
elif network_type == 'blstm':
    # Get lstm cell output
    try:
        output, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        output = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)
else:
    raise Exception('Unknown network! '+network_type)

cost = tf.reduce_sum((output[-1] - y) ** 2)

optim = tf.train.GradientDescentOptimizer(0.01)
train_op = optim.minimize(cost)

config = tf.ConfigProto()
config.inter_op_parallelism_threads = inter_op
config.intra_op_parallelism_threads = intra_op
session = tf.Session(config=config)
#session.run(tf.initialize_all_variables())
session.run(tf.global_variables_initializer())
#print(session.run(tf.report_uninitialized_variables()))
session.run(train_op, feed_dict=get_feed_dict(xinput, ytarget))

print("Setup : compile + forward/backward x 1")
print("--- %s seconds" % (time.time() - start))

start = time.time()
for i in range(0, loops):
    session.run(output[-1], feed_dict=get_feed_dict(xinput))
end = time.time()
print("Forward:")
print("--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (forward_samples, end - start, forward_samples / (end - start), (end - start) / forward_samples))

kernel = (session.run((tf_variables.trainable_variables()[0])))
bias = (session.run((tf_variables.trainable_variables()[1])))


#intelLSTM
print("\n------------intelLSTM------------")
lstm_module = tf.load_op_library('/home/wentingj/tf_bind_rnn/lstm.so')
w_x = tf.transpose(tf.slice(tf_variables.trainable_variables()[0], [0, 0], [input_size, 4*hidden_size]), perm=[1, 0])
w_ix = tf.slice(w_x, [0,0], [hidden_size,input_size])
w_cx = tf.slice(w_x, [hidden_size,0], [hidden_size,input_size])
w_fx = tf.slice(w_x, [hidden_size*2,0], [hidden_size,input_size])
w_ox = tf.slice(w_x, [hidden_size*3,0], [hidden_size,input_size])
w_x_intel = tf.concat([w_fx, w_ix, w_cx, w_ox], 0)

w_h = tf.transpose(tf.slice(tf_variables.trainable_variables()[0], [input_size, 0], [hidden_size, 4*hidden_size]), perm=[1, 0])
w_ih = tf.slice(w_h, [0,0], [hidden_size, hidden_size])
w_ch = tf.slice(w_h, [hidden_size,0], [hidden_size,hidden_size])
w_fh = tf.slice(w_h, [hidden_size*2,0], [hidden_size,hidden_size])
w_oh = tf.slice(w_h, [hidden_size*3,0], [hidden_size,hidden_size])
w_h_intel = tf.concat([w_fh, w_ih, w_ch, w_oh], 0)

b = tf_variables.trainable_variables()[1]
b_i = tf.slice(b, [0], [hidden_size])
b_c = tf.slice(b, [hidden_size], [hidden_size])
b_f = tf.slice(b, [hidden_size*2], [hidden_size])
b_o = tf.slice(b, [hidden_size*3], [hidden_size])
b_intel = tf.concat([b_f, b_i, b_c, b_o], 0)

h_0 = tf.transpose(init_state[0], perm=[1, 0])
c_0 = tf.transpose(init_state[1], perm=[1, 0])
output_intel = lstm_module.intel_lstm(xinput_intel, (w_x_intel), (w_h_intel), (b_intel), (h_0), (c_0))

#print("diff=", session.run((tf.transpose(output_intel, perm=[0, 2, 1]) - output[-1])/output[-1], feed_dict=get_feed_dict(xinput)))
print("diff=", session.run(tf.transpose(output_intel, perm=[0, 2, 1]) - output[-1], feed_dict=get_feed_dict(xinput)))
#start = time.time()
##for i in xrange(0, n_batch):
#for i in range(0, n_batch):
#    session.run(train_op, feed_dict=get_feed_dict(xinput, ytarget))
#end = time.time()
#print("Forward + Backward:")
#print("--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples))
