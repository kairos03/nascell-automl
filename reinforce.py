import tensorflow as tf
import random
import numpy as np

# reinforcement learning for NAS(rnn as controller)
class Reinforce():
    # init 
    def __init__(self, sess, optimizer, policy_network, max_layers, global_step,
                 division_rate=100.0,
                 reg_param=0.001,
                 discount_factor=0.99,
                 exploration=0.3):
        self.sess = sess                        # tf.Session
        self.optimizer = optimizer              # policy optimizer
        self.policy_network = policy_network    # policy network
        self.division_rate = division_rate      # TODO ??
        self.reg_param = reg_param              # regulization lambda
        self.discount_factor=discount_factor    # TODO ??
        self.max_layers = max_layers            # maximum layer
        self.global_step = global_step          # global step

        self.reward_buffer = []                 # reward buffer for policy network
        self.state_buffer = []                  # state buffer for policy network

        self.create_variables()                 # TODO create which varialbes
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)    # get graph variable list
        self.sess.run(tf.variables_initializer(var_lists))              # intialze varialbes
    
    def get_action(self, state):
        """
        get actions from state
        """
        # predict action 
        return self.sess.run(self.predicted_action, {self.states: state})
        # TODO why don't use exploaration???
        if random.random() < self.exploration:
            return np.array([[random.sample(range(1, 35), 4*self.max_layers)]])
        else:
            return self.sess.run(self.predicted_action, {self.states: state})

    def create_variables(self):
        """
        create Controller(reinforce) variables
        """
        # add controller input varialbe as "model_inputs"
        with tf.name_scope("model_inputs"):
            # raw state representation
            self.states = tf.placeholder(tf.float32, [None, self.max_layers*4], name="states")

        # add predict_actions varialbe as "predict_actions"
        with tf.name_scope("predict_actions"):
            # initialize policy network
            with tf.variable_scope("policy_network"):
                # create policy_network
                self.policy_outputs = self.policy_network(self.states, self.max_layers)
            # actcion scores same as policy outputs TODO why? duplicate? 
            self.action_scores = tf.identity(self.policy_outputs, name="action_scores")

            # fianl predicted_cation TODO what is division_rate and why multiply?
            self.predicted_action = tf.cast(tf.scalar_mul(self.division_rate, self.action_scores), tf.int32, name="predicted_action")

        # for regularization loss
        policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")

        # compute loss and gradients
        with tf.name_scope("compute_gradients"):
            # gradients for selecting action from policy network TODO What is discounted_rewards
            self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
            
            # calculate logprob using policy_network already defined above code.
            with tf.variable_scope("policy_network", reuse=True):
                self.logprobs = self.policy_network(self.states, self.max_layers)
                print("self.logprobs", self.logprobs)

            # compute policy loss and regularization loss                                   TODO       v???
            self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logprobs[:, -1, :], labels=self.states)
            self.pg_loss            = tf.reduce_mean(self.cross_entropy_loss)
            self.reg_loss           = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables]) # Regularization
            self.loss               = self.pg_loss + self.reg_param * self.reg_loss

            #compute gradients
            self.gradients = self.optimizer.compute_gradients(self.loss)
            
            # compute policy gradientsu TODO See policy gradient
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (grad * self.discounted_rewards, var)

            # training update
            with tf.name_scope("train_policy_network"):
                # apply gradients to update policy network
                self.train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)

    def storeRollout(self, state, reward):
        """
        store state and reward
        """
        self.reward_buffer.append(reward)
        self.state_buffer.append(state[0])

    def train_step(self, steps_count):
        """
        training controller(reinforce) using policy networks
        """
        # TODO?? division rate??
        # get number of steps_count state from state_buffer 
        states = np.array(self.state_buffer[-steps_count:])/self.division_rate
        # get number of steps_count reward from reward buffer
        rewars = self.reward_buffer[-steps_count:]
        # run training and calculate operations  
        _, ls = self.sess.run([self.train_op, self.loss],
                     {self.states: states,
                      self.discounted_rewards: rewars})
        return ls
