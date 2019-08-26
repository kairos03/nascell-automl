import numpy as np
import tensorflow as tf
import argparse
import datetime

from cnn import CNN
from net_manager import NetManager
from reinforce import Reinforce

from tensorflow.examples.tutorials.mnist import input_data

# get arguments : max layer
def parse_args():
    desc = "TensorFlow implementation of 'Neural Architecture Search with Reinforcement Learning'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--max_layers', default=2)

    args = parser.parse_args()
    args.max_layers = int(args.max_layers)
    return args


# ploicy network
def policy_network(state, max_layers):
    '''
    Policy network is a main network for searching optimal architecture
    it uses NAS - Neural Architecture Search recurrent network cell.
    https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L1363

    Args:
        state: current state of required topology
        max_layers: maximum number of layers
    Returns:
        3-D tensor with new state (new topology)
    '''
    # add name scope
    with tf.name_scope("policy_network"):
        # use nas_cell to predict sequence of actions 
        # num of actions is 4 * max_layer (Rnn Cell Hidden size)
        # `each layer has 4 actions` that [(id, filter_size), cnn_num_filters, max_pool_ksize, dropout_rate] 
        # you can find this to cnn.py:5(first three param): and net_manager.py:21(dropout_rate)
        nas_cell = tf.contrib.rnn.NASCell(4 * max_layers)
        # build rnn model with nas_cell
        # cell = nas_cell
        # inputs = state dim + 1, fit dim as 3
        # and rnn return outputs([batch, max_time, cell.output_size]), state([batch] + cell.state_size)
        outputs, state = tf.nn.dynamic_rnn(
            nas_cell,
            tf.expand_dims(state, -1), # [batch, max_time(T), state] == [1, actions(4)*max_layer, 1]
            dtype=tf.float32
        )
        # make bias and initialize by 0.5
        bias = tf.Variable([0.05]*4*max_layers)
        # add bias to output
        outputs = tf.nn.bias_add(outputs, bias)
        # print outputs, outputs last action, 
        # TODO WHAT IS OUTPUT?
        print("outputs: ", outputs, outputs[:, -1:, :],  tf.slice(outputs, [0, 4*max_layers-1, 0], [1, 1, 4*max_layers]))
        #return tf.slice(outputs, [0, 4*max_layers-1, 0], [1, 1, 4*max_layers]) # Returned last output of rnn
        return outputs[:, -1:, :]      

# training NAS
def train(mnist):
    # use globa varialbe args
    global args
    # create session to run code
    sess = tf.Session()
    # make global_step 
    global_step = tf.Variable(0, trainable=False)
    # intial lerarning rate # TODO NOT USED VARIALBE. MAY BE USE TO LEARING_RATE at below line's first param
    starter_learning_rate = 0.1
    # apply exponential learning rate decay, start_lr=0.99, global_step, decay_step=500, decay_rate=0.96, staircase= True
    # staircase => divide by int 
    learning_rate = tf.train.exponential_decay(0.99, global_step,
                                           500, 0.96, staircase=True)
    # RMSPropOptimizer use above lr
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    # make reinfoce env
    reinforce = Reinforce(sess, optimizer, policy_network, args.max_layers, global_step)
    # network manager for training subnetworks with Reinforcement Learning
    net_manager = NetManager(num_input=784,         # input dim 28x28 mnist
                             num_classes=10,        # number of classes 10 mnist
                             learning_rate=0.001,   # intial learning rate
                             mnist=mnist,           # dataset mnist (tensorflow dataset object)
                             bathc_size=100)        # mini-batch size

    # maximum episodes to training
    MAX_EPISODES = 2500
    # step start from 0 
    step = 0
    # intial state [cnn_filter_size, cnn_filter_num, maxpool_ksize, dropout_rate] * max_layer
    state = np.array([[10.0, 128.0, 1.0, 1.0]*args.max_layers], dtype=np.float32)
    # init previous accuracy and total rewards
    pre_acc = 0.0
    total_rewards = 0

    # run episodes
    for i_episode in range(MAX_EPISODES):
        # get next action from reinforce
        action = reinforce.get_action(state)
        # print action
        print("ca:", action)
        # if actions value is all biger then 0 (valid) get reworkd else...
        if all(ai > 0 for ai in action[0][0]):
            reward, pre_acc = net_manager.get_reward(action, step, pre_acc)
            print("=====>", reward, pre_acc)
        # else reword -1
        else:
            reward = -1.0
        # sum all rewards
        total_rewards += reward

        # In our sample action is equal state
        state = action[0]
        # rollout state. See reinforce code
        reinforce.storeRollout(state, reward)

        # step 
        step += 1
        # train step
        ls = reinforce.train_step(1)
        # logging
        log_str = "current time:  "+str(datetime.datetime.now().time())+" episode:  "+str(i_episode)+" loss:  "+str(ls)+" last_state:  "+str(state)+" last_reward:  "+str(reward)+"\n"
        log = open("lg3.txt", "a+")
        log.write(log_str)
        log.close()
        print(log_str)

def main():
    # get args
    global args
    args = parse_args()

    # read mnist dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # training nas
    train(mnist)

if __name__ == '__main__':
  main()
