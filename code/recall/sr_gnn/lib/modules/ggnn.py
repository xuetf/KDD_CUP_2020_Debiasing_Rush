import tensorflow as tf
from .block import Block


class GGNN(Block):
    def __init__(self, hidden_size=64, batch_size=256, gru_step=1, name=None):
        super().__init__(name)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gru_step = gru_step
        with tf.variable_scope(self._name):
            self.w_in = self._get_variable('w_in', [hidden_size, hidden_size])
            self.b_in = self._get_variable('b_in', [hidden_size])
            self.w_out = self._get_variable('w_out', [hidden_size, hidden_size])
            self.b_out = self._get_variable('b_out', [hidden_size])
            self.gru_cell = tf.nn.rnn_cell.GRUCell(hidden_size)

    def __call__(self, adj_in, adj_out, graph_items_state, node_weight=None):
        """
        :param adj_in:  shape=[batch_size, graph_node_count, graph_node_count]
        :param adj_out: shape=[batch_size, graph_node_count, graph_node_count]
        :param graph_items_state: shape=[batch_size, graph_node_count, input_size]
        :param node_weight: shape=[batch_size,graph_node_count]
        :return: new state, shape=[batch_size, graph_node_count, hidden_size]
        """
        with tf.variable_scope(self._name):
            state = graph_items_state
            for _ in range(self.gru_step):
                state_in = tf.reshape(state, [self.batch_size, -1, self.hidden_size])
                if node_weight is not None:
                    n_w = tf.tile(tf.expand_dims(node_weight, 2), [1, 1, self.hidden_size])
                    state_in = tf.multiply(tf.reshape(state, [self.batch_size, -1, self.hidden_size]), n_w)
                gru_in_in = tf.matmul(tf.reshape(state_in, [-1, self.hidden_size]), self.w_in) + self.b_in
                gru_in_in = tf.reshape(gru_in_in, [self.batch_size, -1, self.hidden_size])
                gru_in_in = tf.matmul(adj_in, gru_in_in)  # batch_size*graph_node_count*hidden_size
                gru_in_out = tf.matmul(tf.reshape(state_in, [-1, self.hidden_size]), self.w_out) + self.b_out
                gru_in_out = tf.reshape(gru_in_out, [self.batch_size, -1, self.hidden_size])
                gru_in_out = tf.matmul(adj_out, gru_in_out)  # batch_size*graph_node_count*hidden_size
                inputs = [gru_in_in, gru_in_out]
                gru_in = tf.concat(inputs, axis=-1)  # batch_size*graph_node_count*(2*hidden_size)
                _, state = self.gru_cell(tf.reshape(gru_in, [-1, 2*self.hidden_size]), tf.reshape(state, [-1, self.hidden_size]))
                state = state + tf.reshape(graph_items_state, [-1, self.hidden_size])
            return tf.reshape(state, [self.batch_size, -1, self.hidden_size])

    def var_list(self):
        return self._vars + self.gru_cell.trainable_variables
