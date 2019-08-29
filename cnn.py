import tensorflow as tf

class CNN():
    """
    CNN network to find
    """
    def __init__(self, num_input, num_classes, cnn_config):
        cnn = [c[0] for c in cnn_config] # listing filter_size
        cnn_num_filters = [c[1] for c in cnn_config]    # listing cnn num of filters
        max_pool_ksize = [c[2] for c in cnn_config]     # listing max pool ksize

        # set input place holder as input_X, shape=[Batch, num_input]
        self.X = tf.placeholder(tf.float32,
                                [None, num_input], 
                                name="input_X")
        self.Y = tf.placeholder(tf.int32, [None, num_classes], name="input_Y") # set input(lable) placeholder as input_Y shape=[batch, num_classes]
        self.dropout_keep_prob = tf.placeholder(tf.float32, [], name="dense_dropout_keep_prob") # set dropout placeholder TODO Keep_prob? what is differente
        self.cnn_dropout_rates = tf.placeholder(tf.float32, [len(cnn), ], name="cnn_dropout_keep_prob") # set cnn dropout rates TODO maybe apply dropout layer by layer

        Y = self.Y  # lable Y
        X = tf.expand_dims(self.X, -1) # input X expand 1 dim
        pool_out = X # TODO why? copy???

        # conv layer
        with tf.name_scope("Conv_part"):
            # each cnn(idd, filter size)
            for idd, filter_size in enumerate(cnn):
                # set ayer id
                with tf.name_scope("L"+str(idd)):
                    # add conv1d layer
                    conv_out = tf.layers.conv1d(
                        pool_out,                           # input
                        filters=cnn_num_filters[idd],       # filters
                        kernel_size=(int(filter_size)),     # kernel
                        strides=1,                          # stride
                        padding="SAME",                     # same padding
                        name="conv_out_"+str(idd),          # name
                        activation=tf.nn.relu,              # activation
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),  # kernle intializer
                        bias_initializer=tf.zeros_initializer   # bias intializer
                    )
                    # maxpool layer
                    pool_out = tf.layers.max_pooling1d(
                        conv_out,                               # input
                        pool_size=(int(max_pool_ksize[idd])),   # pool_size
                        strides=1,                              # strides
                        padding='SAME',                         # same
                        name="max_pool_"+str(idd)               # name
                    )
                    pool_out = tf.nn.dropout(pool_out, self.cnn_dropout_rates[idd]) # dropout 

            flatten_pred_out = tf.contrib.layers.flatten(pool_out)          # flatten
            self.logits = tf.layers.dense(flatten_pred_out, num_classes)    # dense

        self.prediction = tf.nn.softmax(self.logits, name="prediction")                                 # prediction layer (softmax)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=Y, name="loss")  # cross entropy loss
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(Y, 1))                         # for accuract check equal
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")              # accuracy
