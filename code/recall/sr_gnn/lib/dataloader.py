import numpy as np
import math
import logging


class DataLoader:
    def __init__(self, path, shuffle=True, set_label=True, has_header=False,
                 max_len=None, has_uid=False, sq_max_len=None):
        self.set_label = set_label
        self.has_header = has_header
        self.has_uid = has_uid
        self.s_max_len = max_len
        self.sq_max_len = sq_max_len
        self.__load(path)
        self.length = len(self.inputs)
        logging.info('Data Loaded, Length: {}， Max Length: {}'.format(self.length, self.max_len))
        self.shuffle = shuffle

    @property
    def count(self):
        return self.length

    @property
    def session_max_length(self):
        return self.max_len

    def __load(self, path):
        with open(path, 'r') as f:
            inputs, next_item, header, uid = [], [], [], []
            self.max_len = 0
            for line in f:
                s = line.split()
                if self.has_header:
                    header.append(s[0])
                    s = list(map(int, s[1:]))
                else:
                    s = list(map(int, s))
                if self.has_uid:
                    uid.append(s[0])
                    s = s[1:]
                if self.s_max_len:
                    s = s[-self.s_max_len:]
                if self.set_label:
                    inputs.append(s[:len(s) - 1])
                    next_item.append(s[-1])
                    if len(s)-1 > self.max_len:
                        self.max_len = len(s) - 1
                else:
                    inputs.append(s)
                    if len(s) > self.max_len:
                        self.max_len = len(s)

            self.inputs = np.asarray(inputs)
            self.next_item = np.asarray(next_item)
            self.header = np.asarray(header)
            self.uid = np.asarray(uid)

    def get_node_weight(self, node_count):
        nw = np.zeros([node_count+1], np.float32)
        for s in self.inputs:
            for i in s:
                nw[i] += 1
        if self.set_label:
            for i in self.next_item:
                nw[i] += 1
        nw = nw/np.median(nw)
        nw = 1/np.sqrt(nw+1)
        return nw

    def generate_batch(self, batch_size, keep_last_same=False):
        if self.shuffle:
            np.random.seed(1)
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            if self.set_label:
                self.next_item = self.next_item[shuffled_arg]
            if self.has_header:
                self.header = self.header[shuffled_arg]
            if self.has_uid:
                self.uid = self.uid[shuffled_arg]
        n_batch = math.ceil(self.length / batch_size)
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        if self.length % batch_size != 0:
            slices[-1] = np.arange((n_batch - 1) * batch_size, self.length)
        if keep_last_same:
            slices[-1] = np.arange(self.length - batch_size, self.length)
            if self.length < batch_size:
                slices[-1] = np.concatenate([np.arange(self.length), np.zeros([batch_size - self.length], int)])
        return slices

    def get_slice(self, index):
        graph_items, adj_in, adj_out, last_node_id, node_position = [], [], [], [], []
        max_n_node = np.max([len(np.unique(u_input)) for u_input in self.inputs[index]])

        for u_input in self.inputs[index]:
            node = np.unique(u_input)
            node_pos = {p: i for i, p in enumerate(node)}
            u_A = np.zeros((max_n_node, max_n_node))
            if self.sq_max_len is not None:
                if len(u_input) < self.sq_max_len:
                    pos = dict(zip(u_input, range(len(u_input), 0, -1)))  # session position
                else:
                    r = [self.sq_max_len]*(len(u_input)-self.sq_max_len)+list(range(self.sq_max_len, 0, -1))
                    pos = dict(zip(u_input, r))  # session position
                node_position.append([pos[v] for v in node] + [0] * (max_n_node - len(node)))

            for i in np.arange(len(u_input) - 1):
                u_A[node_pos[u_input[i]]][node_pos[u_input[i + 1]]] = 1

            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in).transpose()
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out).transpose()

            adj_in.append(u_A_in)
            adj_out.append(u_A_out)
            graph_items.append(node.tolist() + [0] * (max_n_node - len(node)))
            last_node_id.append(np.where(node == u_input[-1])[0][0])
        attr_dict = {}
        if self.set_label:
            attr_dict['next_item'] = self.next_item[index]
        if self.has_header:
            attr_dict['header'] = self.header[index]
        if self.has_uid:
            attr_dict['uid'] = self.uid[index]
        if self.sq_max_len is not None:
            attr_dict['node_pos'] = node_position
        return adj_in, adj_out, graph_items, last_node_id, attr_dict
