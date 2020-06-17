from gensim.models.word2vec import *
from ..conf import *
import tensorflow as tf
import numpy as np
import pandas as pd
from ..global_variables import *


def get_word2vec_feat(full_user_item_df):
    import time
    seq_list = full_user_item_df['hist_item_id'].apply(lambda x: [str(i) for i in x]).values
    print(seq_list.shape)
    begin_time = time.time()
    model = Word2Vec(seq_list, size=w2v_dim, window=5, min_count=0, workers=40, sg=0, hs=1)
    end_time = time.time()
    run_time = end_time - begin_time

    print('word2vec time：', round(run_time, 2))  # 该循环程序运行时间： 1.4201874732

    vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
    word2vec_item_embed_dict = dict(vocab_list)

    return word2vec_item_embed_dict, {}


def sparse_feat_fit(total_click):
    from sklearn.preprocessing import LabelEncoder
    # sparse features one-hot
    feat_lbe_dict = {}
    for feat in sparse_feat:
        if feat in time_feat: continue
        lbe = LabelEncoder()
        lbe.fit(total_click[feat].astype(str))
        feat_lbe_dict[feat] = lbe

    item_raw_id2_idx_dict = dict(zip(feat_lbe_dict['item_id'].classes_,
                                     feat_lbe_dict['item_id'].transform(
                                         feat_lbe_dict['item_id'].classes_) + 1, ))  # get dictionary
    user_raw_id2_idx_dict = dict(zip(feat_lbe_dict['user_id'].classes_,
                                     feat_lbe_dict['user_id'].transform(
                                         feat_lbe_dict['user_id'].classes_) + 1, ))  # get dictionary
    set_glv('feat_lbe_dict', feat_lbe_dict)
    set_glv('item_raw_id2_idx_dict', item_raw_id2_idx_dict)
    set_glv('user_raw_id2_idx_dict', user_raw_id2_idx_dict)




def sparse_feat_transform(df):
    df['hist_item_id'] = df['hist_item_id'].apply(lambda seq: [item_raw_id2_idx_dict[str(x)] for x in seq])

    for hist_id in var_len_feat:
        df[hist_id] = tf.keras.preprocessing.sequence.pad_sequences(df[hist_id],
                                                                    value=0, maxlen=max_seq_len, truncating='pre',
                                                                    padding='post').tolist()
    for feat in sparse_feat:
        print(feat)
        if feat in time_feat: continue
        df[feat] = feat_lbe_dict[feat].transform(df[feat].astype(str)) + 1
    return df


def fillna(df, sparse_feat, dense_feat):
    for sp in sparse_feat:
        df[sp].fillna('-1', inplace=True)

    for ds in dense_feat:
        df[ds].fillna(0.0, inplace=True)  # all_click_user_item_df[ds].mean()
    return df


def organize_user_item_feat(df, item_feat_df, sparse_feat, dense_feat,
                            is_interest=True, is_w2v=False, word2vec_item_embed_dict=None):
    item_content_vec_dict = get_glv('item_content_vec_dict')

    full_user_item_df = pd.merge(df, item_feat_df, how='left', on='item_id')
    full_user_item_df = fillna(full_user_item_df, sparse_feat, dense_feat)
    print('origin data done')

    if is_interest:
        # history interest
        full_user_item_df = obtain_user_hist_interest_feat(full_user_item_df, item_content_vec_dict)
        print('interest done')

    if is_w2v:
        organize_word2vec_feat(full_user_item_df, word2vec_item_embed_dict)
        print('word2vec done')

    full_user_item_df = sparse_feat_transform(full_user_item_df)

    return full_user_item_df


def obtain_user_hist_interest_feat(full_user_item_df, item_vec_dict):
    item_content_vec_dict = get_glv('item_content_vec_dict')

    def weighted_agg_content(hist_item_id_list):

        weighted_content = np.zeros(128 * 2)
        hist_num = len(hist_item_id_list)
        for loc, i in enumerate(hist_item_id_list):
            loc_weight = (0.9 ** (hist_num - loc))
            if i in item_vec_dict:
                weighted_content += loc_weight * item_vec_dict[i]
        return weighted_content

    user_interest_vec = full_user_item_df['hist_item_id'].apply(weighted_agg_content).tolist()
    user_interest_df = pd.DataFrame(user_interest_vec, columns=['interest_' + col for col in item_dense_feat])

    full_user_item_df[user_interest_df.columns] = user_interest_df

    # begin compute degree
    target_item_vec = full_user_item_df[item_dense_feat].values
    user_interest_vec = np.array(user_interest_vec)

    txt_interest_degree_array = target_item_vec[:, 0:128] * user_interest_vec[:, 0:128]
    txt_interest_degree_list = np.sum(txt_interest_degree_array, axis=1)
    full_user_item_df['txt_interest_degree'] = txt_interest_degree_list.tolist()

    img_interest_degree_array = target_item_vec[:, 128:] * user_interest_vec[:, 128:]
    img_interest_degree_list = np.sum(img_interest_degree_array, axis=1)
    full_user_item_df['img_interest_degree'] = img_interest_degree_list.tolist()

    full_user_item_df['interest_degree'] = full_user_item_df['img_interest_degree'] + full_user_item_df[
        'img_interest_degree']

    for f in ['interest_' + col for col in item_dense_feat] + ['img_interest_degree', 'img_interest_degree',
                                                               'interest_degree']:
        full_user_item_df[f].fillna(0.0, inplace=True)
    print('obtain user dynamic feat done')

    def hist_2_target_cnt(hist_target_item_list, hist_no):
        target_item = hist_target_item_list[-1]
        if target_item not in item_content_vec_dict:
            return [0.0, 0.0, 0.0]

        hist_target_item_list = hist_target_item_list[: -1]

        if len(hist_target_item_list) >= hist_no:
            hist_item = hist_target_item_list[-hist_no]
            if hist_item in item_content_vec_dict:
                txt_cnt_sim = np.dot(item_content_vec_dict[target_item][0:128], item_content_vec_dict[hist_item][0:128])
                img_cnt_sim = np.dot(item_content_vec_dict[target_item][128:], item_content_vec_dict[hist_item][128:])
                return txt_cnt_sim, img_cnt_sim, txt_cnt_sim + img_cnt_sim

        return [0.0, 0.0, 0.0]

    hist_target_items_series = full_user_item_df['hist_item_id'] + full_user_item_df['item_id'].apply(lambda x: [x])
    full_user_item_df['txt_cnt_sim_last_1'], full_user_item_df['img_cnt_sim_last_1'], full_user_item_df[
        'cnt_sim_last_1'] = zip(*hist_target_items_series.apply(lambda x: hist_2_target_cnt(x, 1)))
    full_user_item_df['txt_cnt_sim_last_2'], full_user_item_df['img_cnt_sim_last_2'], full_user_item_df[
        'cnt_sim_last_2'] = zip(*hist_target_items_series.apply(lambda x: hist_2_target_cnt(x, 2)))
    full_user_item_df['txt_cnt_sim_last_3'], full_user_item_df['img_cnt_sim_last_3'], full_user_item_df[
        'cnt_sim_last_3'] = zip(*hist_target_items_series.apply(lambda x: hist_2_target_cnt(x, 3)))

    def hist_2_target_time_diff(hist_time_list, hist_num=3):
        target_time = hist_time_list[-1]
        hist_time_list = hist_time_list[: -1]

        hist_time_diff = []
        for hist_time in hist_time_list[::-1][0:hist_num]:
            diff_time = target_time - hist_time
            hist_time_diff.append(diff_time)

        while len(hist_time_diff) != hist_num:
            hist_time_diff.append(0.1)

        return hist_time_diff

    hist_target_time_series = full_user_item_df['hist_time'] + full_user_item_df['time'].apply(lambda x: [x])
    full_user_item_df['time_diff_1'], full_user_item_df['time_diff_2'], full_user_item_df['time_diff_3'] = zip(
        *hist_target_time_series.apply(hist_2_target_time_diff))

    return full_user_item_df


def organize_word2vec_feat(full_user_item_df, w2v_item_embed_dict):
    def lookup_item_word2vec_embed(item_id):
        return w2v_item_embed_dict.get(str(item_id), np.zeros(w2v_dim)).tolist()

    def hist_2_target_w2v(hist_target_item_list, hist_no):
        target_item = hist_target_item_list[-1]
        if str(target_item) not in w2v_item_embed_dict:
            return 0.0

        hist_target_item_list = hist_target_item_list[: -1]

        if len(hist_target_item_list) >= hist_no:
            hist_item = hist_target_item_list[-hist_no]
            if str(hist_item) in w2v_item_embed_dict:
                w2v_sim = np.dot(w2v_item_embed_dict[str(target_item)], w2v_item_embed_dict[str(hist_item)])
                return w2v_sim
        return 0.0

    hist_target_items_series = full_user_item_df['hist_item_id'] + full_user_item_df['item_id'].apply(lambda x: [x])
    full_user_item_df['w2v_sim_last_1'] = hist_target_items_series.apply(lambda x: hist_2_target_w2v(x, 1))
    full_user_item_df['w2v_sim_last_2'] = hist_target_items_series.apply(lambda x: hist_2_target_w2v(x, 2))
    full_user_item_df['w2v_sim_last_3'] = hist_target_items_series.apply(lambda x: hist_2_target_w2v(x, 3))

    item_w2v_embed_list = full_user_item_df['item_id'].apply(lookup_item_word2vec_embed).tolist()  # target_item_id
    item_w2v_cols = ['item_w2v_embed_{}'.format(i) for i in range(w2v_dim)]
    item_w2v_pd = pd.DataFrame(item_w2v_embed_list, columns=item_w2v_cols)

    full_user_item_df[item_w2v_cols] = item_w2v_pd

    return full_user_item_df
