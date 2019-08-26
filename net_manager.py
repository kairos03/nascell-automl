import tensorflow as tf
from cnn import CNN


class NetManager():
    """
    Network manager for training 
    """
    def __init__(self, num_input, num_classes, learning_rate, mnist,
                 max_step_per_action=5500*3,
                 bathc_size=100,
                 dropout_rate=0.85):

        """
        params:
            number of input
            number of classes
            initial leraning rate
            mnist dataset
            max step per action
            mini-batch size
            dropout rate(last dense later)
        """
        self.num_input = num_input
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.mnist = mnist

        self.max_step_per_action = max_step_per_action
        self.bathc_size = bathc_size
        self.dropout_rate = dropout_rate

    def get_reward(self, action, step, pre_acc):
        """
        get reward 

        params:
            self: self object
            action: action
            step: step
            pre_acc: previous accuracy
        """
        # split action by each layer
        action = [action[0][0][x:x+4] for x in range(0, len(action[0][0]), 4)]
        # listing cnn layer dropout rate
        cnn_drop_rate = [c[3] for c in action]
        # with graph
        with tf.Graph().as_default() as g:
            # graph container TODO what is this
            with g.container('experiment'+str(step)):
                # create model using action
                model = CNN(self.num_input, self.num_classes, action)
                # loss
                loss_op = tf.reduce_mean(model.loss)
                # CNN optimizer
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                # set loss to cnn optimizer
                train_op = optimizer.minimize(loss_op)

                # session
                with tf.Session() as train_sess:
                    # init global variables
                    init = tf.global_variables_initializer()
                    train_sess.run(init)

                    # training step to max_step_per_cations
                    for step in range(self.max_step_per_action):
                        # get mini-batch
                        batch_x, batch_y = self.mnist.train.next_batch(self.bathc_size)
                        # training model
                        feed = {model.X: batch_x,
                                model.Y: batch_y,
                                model.dropout_keep_prob: self.dropout_rate,
                                model.cnn_dropout_rates: cnn_drop_rate}
                        _ = train_sess.run(train_op, feed_dict=feed)

                        # calculate batch loss and accuracy each 100 steps and print
                        if step % 100 == 0:
                            # Calculate batch loss and accuracy
                            loss, acc = train_sess.run(
                                [loss_op, model.accuracy],
                                feed_dict={model.X: batch_x,
                                           model.Y: batch_y,
                                           model.dropout_keep_prob: 1.0,
                                           model.cnn_dropout_rates: [1.0]*len(cnn_drop_rate)})
                            print("Step " + str(step) +
                                  ", Minibatch Loss= " + "{:.4f}".format(loss) +
                                  ", Current accuracy= " + "{:.3f}".format(acc))
                    # after training test mini-batch
                    batch_x, batch_y = self.mnist.test.next_batch(10000)
                    # run test
                    loss, acc = train_sess.run(
                                [loss_op, model.accuracy],
                                feed_dict={model.X: batch_x,
                                           model.Y: batch_y,
                                           model.dropout_keep_prob: 1.0,
                                           model.cnn_dropout_rates: [1.0]*len(cnn_drop_rate)})
                    # print actual accuracy
                    print("!!!!!!acc:", acc, pre_acc)
                    # return minial reward and accuracy TODO ?? why use minial reward
                    if acc - pre_acc <= 0.01:
                        return acc, acc 
                    else:
                        return 0.01, acc
                    
