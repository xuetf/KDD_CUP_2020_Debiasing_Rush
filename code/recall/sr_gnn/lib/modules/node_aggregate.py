import tensorflow as tf
import math
from .block import Block


class NodeAggregate(Block):
    def __init__(self, hidden_size=64, batch_size=256, name=None):
        super().__init__(name)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.var_init = 1.0 / math.sqrt(self.hidden_size)

    def __call__(self, graph_items_state, mask, last_item_node_id, **kwargs):
        """
        :param graph_items_state: shape=[batch_size, graph_node_count, hidden_size]
        :param mask: graph_item mask, 1 for valid node, 0 for nonexistent node, shape=[batch_size, graph_node_count]
        :param last_item_node_id: last node id of session, [batch_size]
        :return: after aggregate: shape=[batch_size, hidden_size]
        """
        raise NotImplementedError


class NodeMELast(NodeAggregate):
    def __init__(self, hidden_size=64, batch_size=256, name=None):
        super().__init__(hidden_size, batch_size, name)
        with tf.variable_scope(self._name):
            self.w = self._get_variable('w', [])

    def __call__(self, graph_items_state, mask, last_item_node_id, **kwargs):
        mask_last_node = tf.one_hot(last_item_node_id, tf.shape(mask)[-1], 0., 1., -1, tf.float32)  # batch_size * graph_node_count
        mask_without_last = tf.multiply(mask_last_node, mask)
        mask_y = tf.tile(tf.expand_dims(mask_without_last, 2), [1, 1, tf.shape(graph_items_state)[-1]])
        sum_states = tf.reduce_sum(tf.multiply(graph_items_state, mask_y), 1)
        c = tf.expand_dims(tf.reduce_sum(mask_without_last, 1)+1e-8, 1)
        avg = sum_states / c
        last_state = tf.gather_nd(graph_items_state, tf.stack([tf.range(self.batch_size), last_item_node_id], axis=1))
        out = avg * self.w + (1 - self.w) * last_state
        return out
