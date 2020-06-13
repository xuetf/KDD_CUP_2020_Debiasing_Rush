import tensorflow as tf
import math
import logging
import numpy as np
import modules
logger = logging.getLogger("model")
tf.set_random_seed(1)


class Model:
    def __init__(self, node_count, restore_path=None, **kwargs):
        self.l2 = kwargs.get('l2', 1e-5)
        self.lr = kwargs.get('lr', 0.001)
        self.hidden_size = kwargs.get('hidden_size', 64)
        self.var_init = 1.0 / math.sqrt(self.hidden_size)
        self.sigma = kwargs.get('sigma', 8)
        self.sq_max_length = kwargs.get('sq_max_len', None)

        if self.sq_max_length is not None:
            self.item_position = tf.placeholder(tf.int32, name="item_position")  # batch_size, None
            self.position_embedding = tf.get_variable("position_embedding", shape=[self.sq_max_length+1, self.hidden_size],
                                                      dtype=tf.float32, initializer=tf.random_uniform_initializer(-self.var_init, self.var_init))

        self.batch_size = tf.placeholder(tf.int32)

        node_weight = kwargs.get('node_weight', None)
        self.node_weight = None
        if node_weight is not None:
            if isinstance(node_weight, str):
                nw = np.load(node_weight)
            else:
                nw = node_weight
            if kwargs.get('node_weight_trainable', False):
                self.node_weight = tf.get_variable('node_weight', [nw.shape[0]], dtype=tf.float32,
                                                   initializer=tf.constant_initializer(nw))
            else:
                self.node_weight = tf.constant(nw, dtype=tf.float32)

        if kwargs.get('feature_init', None) is not None:
            init = tf.constant_initializer(np.load(kwargs['feature_init']))
            logger.info("Use Feature Init")
        else:
            init = tf.random_uniform_initializer(-self.var_init, self.var_init)

        self.node_embedding = (tf.get_variable("node_embedding",
                                               shape=[node_count, self.hidden_size], dtype=tf.float32,
                                               initializer=init))
        self.next_item = tf.placeholder(tf.int32)  # batch_size
        self.adj_in = tf.placeholder(tf.float32)  # batch_size, None, None
        self.adj_out = tf.placeholder(tf.float32)  # batch_size, None, None
        self.graph_item = tf.placeholder(tf.int32)  # batch_size, None
        self.last_item_node_id = tf.placeholder(tf.int32)  # batch_size
        self.ggnn = modules.GGNN(self.hidden_size, self.batch_size, kwargs.get('gru_step', 1))
        self.node_aggregator = modules.NodeMELast(self.hidden_size, self.batch_size)

        self.loss, self.session_state, self.logits = self.__forward()

        self.global_step = tf.train.get_or_create_global_step()
        params = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=params)
        with open('vars', 'w') as f:
            for p in params:
                f.write(str(p)+'\n')

        lr_dc = kwargs.get('lr_dc', None)
        dc_rate = kwargs.get('dc_rate', 0.9)
        self.learning_rate = tf.train.exponential_decay(self.lr, global_step=self.global_step, decay_steps=lr_dc,
                                                        decay_rate=dc_rate, staircase=True) if lr_dc else self.lr
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.loss_train = self.loss + tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.l2
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = opt.minimize(self.loss_train, global_step=self.global_step)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        if restore_path is not None:
            try:
                self.saver.restore(self.sess, restore_path)
            except ValueError as ve:
                logger.info(ve)
            except:
                logger.info("Restore Failed")

    def __forward(self):
        node_state = tf.nn.embedding_lookup(self.node_embedding, self.graph_item)
        node_state = tf.nn.l2_normalize(node_state, -1)
        graph_items_mask = tf.cast(tf.greater(self.graph_item, tf.zeros_like(self.graph_item)), tf.float32)
        nw = None
        if self.node_weight is not None:
            nw = tf.nn.embedding_lookup(self.node_weight, self.graph_item)

        node_state = self.ggnn(self.adj_in, self.adj_out, node_state, nw)
        if self.sq_max_length is not None:
            position_state = tf.nn.embedding_lookup(self.position_embedding, self.item_position)
            node_state = node_state + position_state
        state = self.node_aggregator(node_state, graph_items_mask, self.last_item_node_id)
        logits = tf.matmul(tf.nn.l2_normalize(state, 1), tf.nn.l2_normalize(self.node_embedding[1:], 1),
                           transpose_b=True) * self.sigma
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.next_item - 1, logits=logits))
        return loss, state, logits

    def __build_feed(self, session, has_next_item=True):
        feed_dict = {}
        if has_next_item:
            adj_in, adj_out, graph_item, last_node_id, next_item = session
        else:
            adj_in, adj_out, graph_item, last_node_id = session
        feed_dict[self.adj_in] = adj_in
        feed_dict[self.adj_out] = adj_out
        feed_dict[self.graph_item] = graph_item
        feed_dict[self.last_item_node_id] = last_node_id
        if has_next_item:
            feed_dict[self.next_item] = next_item
        return feed_dict

    def run_train(self, input_session, item_pos=None):
        """
        :param item_pos
        :param input_session: (adj_in, adj_out, graph_item, last_node_id, next_item)
        :return: train_loss
        """
        node_pos_dict = {} if self.sq_max_length is None else {self.item_position: item_pos}
        feed_dict = {self.batch_size: len(input_session[2]), **node_pos_dict, **self.__build_feed(input_session)}
        _, train_loss = self.sess.run([self.opt, self.loss_train], feed_dict=feed_dict)
        return train_loss

    def run_eval(self, input_session, item_pos=None):
        node_pos_dict = {} if self.sq_max_length is None else {self.item_position: item_pos}
        feed_dict = {self.batch_size: len(input_session[2]), **node_pos_dict, **self.__build_feed(input_session)}
        return self.sess.run([self.loss, self.logits], feed_dict=feed_dict)

    def run_embedding(self):
        return self.sess.run(self.node_embedding)

    def run_session_embedding(self, input_session):
        feed_dict = {self.batch_size: len(input_session[2]),
                     **self.__build_feed(input_session)}
        return self.sess.run(self.session_state, feed_dict=feed_dict)

    def run_predict(self, input_session, item_pos=None):
        node_pos_dict = {} if self.sq_max_length is None else {self.item_position: item_pos}
        feed_dict = {self.batch_size: len(input_session[2]), **node_pos_dict, **self.__build_feed(input_session, False)}
        return self.sess.run(self.logits, feed_dict=feed_dict)

    def run_step(self):
        return self.sess.run(self.global_step)

    def save(self, path, global_step=None):
        self.saver.save(self.sess, path, global_step)



