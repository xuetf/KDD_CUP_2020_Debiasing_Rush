import collections
import tensorflow as tf
import math

_LAYER_UIDS = collections.defaultdict(lambda: 0)


def get_layer_uid(layer_name=''):
    _LAYER_UIDS[layer_name] += 1
    return _LAYER_UIDS[layer_name]


class Block(object):
    def __init__(self, name=None):
        if name is None:
            layer_name = self.__class__.__name__.lower()
            name = layer_name + '_' + str(get_layer_uid(layer_name))
        self._name = name
        self._vars = []

    def _get_variable(self, name, shape, dtype=None, initializer=None):
        if dtype is None:
            dtype = tf.float32
        if initializer is None:
            var_init = 1.0 / math.sqrt(shape[0] if len(shape) > 0 else 1)
            initializer = tf.random_uniform_initializer(-var_init, var_init)
        v = tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer)
        if v not in self._vars:
            self._vars.append(v)
        return v

    def var_list(self):
        return self._vars
