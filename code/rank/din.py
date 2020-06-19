"""
Author:
    Taofeng Xue, xuetfchn@foxmail.com
Reference:
    DeepCTR, Easy-to-use,Modular and Extendible package of deep-learning based CTR models, https://github.com/shenweichen/DeepCTR
"""


from ..process.load_data import *
from ..process.recommend_process import *
from tensorflow.python.keras.layers import Dense,Concatenate

from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.sequence import AttentionSequencePoolingLayer
from deepctr.layers.utils import concat_func, NoMask
from tensorflow.python.keras.initializers import RandomNormal, Constant
from deepctr.inputs import build_input_features,create_embedding_matrix,SparseFeat,VarLenSparseFeat,DenseFeat,embedding_lookup,get_dense_input,varlen_embedding_lookup,get_varlen_pooling_list,combined_dnn_input
from tensorflow.python.keras.layers import Embedding, Flatten
from deepctr.inputs import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat
from tensorflow.python.keras.models import Model
import tensorflow as tf
tf.set_random_seed(1234)
from ..global_variables import *


def get_init_item_embed():
    global item_embed_np
    item_raw_id2_idx_dict = get_glv('item_raw_id2_idx_dict')
    item_content_vec_dict = get_glv('item_content_vec_dict')

    item_cnt = len(item_raw_id2_idx_dict)
    item_embed_np = np.zeros((item_cnt + 1, 256))
    for raw_id, idx in item_raw_id2_idx_dict.items():
        vec = item_content_vec_dict[int(raw_id)]
        item_embed_np[idx, :] = vec
    return item_embed_np


def get_init_user_embed(target_phase, is_use_whole_click=True):
    user_raw_id2_idx_dict = get_glv('user_raw_id2_idx_dict')
    item_content_vec_dict = get_glv('item_content_vec_dict')

    global user_embed_np
    all_click, click_q_time = get_phase_click(target_phase)
    if is_use_whole_click:
        phase_click = get_whole_phase_click(all_click, click_q_time)
    else:
        phase_click = all_click

    user_item_time_hist_dict = get_user_item_time_dict(phase_click)

    def weighted_agg_content(hist_item_id_list):
        weighted_vec = np.zeros(128*2)
        hist_num = len(hist_item_id_list)
        sum_weight = 0.0
        for loc, (i,t) in enumerate(hist_item_id_list):
            loc_weight = (0.9**(hist_num-loc))
            if i in item_content_vec_dict:
                sum_weight += loc_weight
                weighted_vec += loc_weight*item_content_vec_dict[i]
        if sum_weight != 0:
            weighted_vec /= sum_weight
            txt_item_feat_np = weighted_vec[0:128] / np.linalg.norm(weighted_vec[0:128])
            img_item_feat_np = weighted_vec[128:] / np.linalg.norm(weighted_vec[128:])
            weighted_vec = np.concatenate([txt_item_feat_np,  img_item_feat_np])
        else:
            print('zero weight...')
        return weighted_vec
    user_cnt = len(user_raw_id2_idx_dict)
    user_embed_np = np.zeros((user_cnt+1, 256))
    for raw_id, idx in user_raw_id2_idx_dict.items():
        if int(raw_id) in user_item_time_hist_dict:
            hist = user_item_time_hist_dict[int(raw_id)]
            vec = weighted_agg_content(hist)
            user_embed_np[idx, :] = vec
    return user_embed_np


def kdd_create_embedding_matrix(feature_columns, l2_reg, init_std, seed, prefix="", seq_mask_zero=True):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = kdd_create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, init_std, seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict


def kdd_create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, init_std, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    sparse_embedding = {}
    for feat in sparse_feature_columns:
        embed_initializer = RandomNormal(mean=0.0, stddev=init_std, seed=seed)
        if feat.embedding_name == 'user_id':
            print('init user embed')
            embed_initializer = Constant(user_embed_np)
        if feat.embedding_name == 'item_id':
            print('init item embed')
            embed_initializer = Constant(item_embed_np)
        sparse_embedding[feat.embedding_name] = Embedding(feat.vocabulary_size, feat.embedding_dim,
                                                                       embeddings_initializer=embed_initializer,
                                                                       name=prefix + '_emb_' + feat.embedding_name)

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            embed_initializer = RandomNormal(mean=0.0, stddev=init_std, seed=seed)
            if feat.embedding_name == 'user_id':
                print('init user embed')
                embed_initializer = Constant(user_embed_np)
            if feat.embedding_name == 'item_id':
                print('init item embed')
                embed_initializer = Constant(item_embed_np)
            sparse_embedding[feat.embedding_name] = Embedding(feat.vocabulary_size, feat.embedding_dim,
                                                              embeddings_initializer=embed_initializer,
                                                              name=prefix + '_seq_emb_' + feat.name,
                                                              mask_zero=seq_mask_zero)
    return sparse_embedding


def KDD_DIN(dnn_feature_columns, history_feature_list, dnn_use_bn=False,
        dnn_hidden_units=(200, 80), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="dice",
        att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024,
        task='binary'):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """

    features = build_input_features(dnn_feature_columns)

    sparse_feature_columns = list(filter(lambda x:isinstance(x,SparseFeat),dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    inputs_list = list(features.values())

    embedding_dict = kdd_create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, init_std, seed, prefix="")

    query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns, history_feature_list,
                                      history_feature_list,to_list=True)
    keys_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns, history_fc_names,
                                     history_fc_names,to_list=True)
    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list,to_list=True)
    dense_value_list = get_dense_input(features, dense_feature_columns)

    sequence_embed_dict = varlen_embedding_lookup(embedding_dict,features,sparse_varlen_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,to_list=True)

    dnn_input_emb_list += sequence_embed_list

    keys_emb = concat_func(keys_emb_list, mask=True)
    deep_input_emb = concat_func(dnn_input_emb_list)
    query_emb = concat_func(query_emb_list, mask=True)
    hist = AttentionSequencePoolingLayer(att_hidden_size, att_activation,
                                         weight_normalization=att_weight_normalization, supports_masking=True)([
        query_emb, keys_emb])

    deep_input_emb = Concatenate()([NoMask()(deep_input_emb), hist])
    deep_input_emb = Flatten()(deep_input_emb)
    dnn_input = combined_dnn_input([deep_input_emb],dense_value_list)
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                 dnn_dropout, dnn_use_bn, seed)(dnn_input)
    final_logit = Dense(1, use_bias=False)(output)

    output = PredictionLayer(task)(final_logit)

    model = Model(inputs=inputs_list, outputs=output)
    return model


HIDDEN_SIZE = (128, 128)
BATCH_SIZE = 1024
EPOCH = 1
EMBED_DIM = 256
TIME_EMBED_DIM = 16
tf.set_random_seed(1234)


def generate_din_feature_columns(data, sparse_features, dense_features):
    feat_lbe_dict = get_glv('feat_lbe_dict')

    sparse_feature_columns = [
        SparseFeat(feat, vocabulary_size=len(feat_lbe_dict[feat].classes_) + 1, embedding_dim=EMBED_DIM)
        for i, feat in enumerate(sparse_features) if feat not in time_feat]

    dense_feature_columns = [DenseFeat(feat, 1, ) for feat in dense_features]

    var_feature_columns = [
        VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=len(feat_lbe_dict['item_id'].classes_) + 1,
                                    embedding_dim=EMBED_DIM, embedding_name='item_id'),
                         maxlen=max_seq_len)]

    # DNN side
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns + var_feature_columns
    # FM side
    linear_feature_columns = sparse_feature_columns + dense_feature_columns + var_feature_columns
    # all feature names
    feature_names = get_feature_names(dnn_feature_columns + linear_feature_columns)

    return feature_names, linear_feature_columns, dnn_feature_columns


def din_main(target_phase, train_final_df, val_final_df=None):
    print('din begin...')
    get_init_item_embed()
    get_init_user_embed(target_phase, is_use_whole_click=True)
    feature_names, linear_feature_columns, dnn_feature_columns = generate_din_feature_columns(train_final_df,
                                                                                              ['user_id',
                                                                                               'item_id'],
                                                                                              dense_features=item_dense_feat + sim_dense_feat + hist_time_diff_feat + hist_cnt_sim_feat + user_interest_dense_feat)
    train_input = {name: np.array(train_final_df[name].values.tolist()) for name in feature_names}
    train_label = train_final_df['label'].values
    if mode == 'offline':
        val_input = {name: np.array(val_final_df[name].values.tolist()) for name in feature_names}
        val_label = val_final_df['label'].values

    EPOCH = 1
    behavior_feature_list = ['item_id']
    model = KDD_DIN(dnn_feature_columns, behavior_feature_list, dnn_hidden_units=HIDDEN_SIZE,
                    att_hidden_size=(128, 64), att_weight_normalization=True,
                    dnn_dropout=0.5)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4), loss="binary_crossentropy",
                  metrics=['binary_crossentropy', tf.keras.metrics.AUC()], )

    if mode == 'offline':
        model.fit(train_input, train_label, batch_size=BATCH_SIZE, epochs=EPOCH,
                  verbose=1, validation_data=(val_input, val_label), )  # 1:20目前最优结果, epoch. 0.8728
    else:
        model.fit(train_input, train_label, batch_size=BATCH_SIZE, epochs=EPOCH,
                  verbose=1)
    return model, feature_names
