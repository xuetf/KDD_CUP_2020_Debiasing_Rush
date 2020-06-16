import numpy as np
from collections import defaultdict
import math
from tqdm import tqdm
import pandas as pd
from ..global_variables import *


def make_user_time_tuple(group_df, user_col='user_id', item_col='item_id', time_col='time'):
    user_time_tuples = list(zip(group_df[user_col], group_df[time_col]))
    return user_time_tuples


def make_item_time_tuple(group_df, user_col='user_id', item_col='item_id', time_col='time'):
    # group_df = group_df.drop_duplicates(subset=[user_col, item_col], keep='last')
    item_time_tuples = list(zip(group_df[item_col], group_df[time_col]))
    return item_time_tuples


def get_user_item_time_dict(df, user_col='user_id', item_col='item_id', time_col='time', is_drop_duplicated=False):
    user_item_ = df.sort_values(by=[user_col, time_col])

    if is_drop_duplicated:
        print('drop duplicates...')
        user_item_ = user_item_.drop_duplicates(subset=['user_id', 'item_id'], keep='last')

    user_item_ = user_item_.groupby(user_col).apply(
        lambda group: make_item_time_tuple(group, user_col, item_col, time_col)).reset_index().rename(
        columns={0: 'item_id_time_list'})
    user_item_time_dict = dict(zip(user_item_[user_col], user_item_['item_id_time_list']))
    return user_item_time_dict


def get_item_user_time_dict(df, user_col='user_id', item_col='item_id', time_col='time'):
    item_user_df = df.sort_values(by=[item_col, time_col])
    item_user_df = item_user_df.groupby(item_col).apply(
        lambda group: make_user_time_tuple(group, user_col, item_col, time_col)).reset_index().rename(
        columns={0: 'user_id_time_list'})
    item_user_time_dict = dict(zip(item_user_df[item_col], item_user_df['user_id_time_list']))
    return item_user_time_dict


def get_user_item_dict(df, user_col='user_id', item_col='item_id', time_col='time'):
    user_item_ = df.groupby(user_col)[item_col].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))
    return user_item_dict


def get_user_min_time_dict(df, user_col='user_id', item_col='item_id', time_col='time'):
    df = df.sort_values(by=[user_col, time_col])
    df = df.groupby(user_col).head(1)
    user_min_time_dict = dict(zip(df[user_col], df[time_col]))
    return user_min_time_dict


def item_based_recommend(sim_item_corr, user_item_time_dict, user_id, top_k, item_num, alpha=15000,
                         item_cnt_dict=None, user_cnt_dict=None, adjust_type='xtf_v0'):
    item_content_sim_dict = get_glv('item_content_sim_dict')
    rank = {}
    if user_id not in user_item_time_dict:
        return []
    interacted_item_times = user_item_time_dict[user_id]
    min_time = min([time for item, time in interacted_item_times])
    interacted_items = set([item for item, time in interacted_item_times])

    miss_item_num = 0
    for loc, (i, time) in enumerate(interacted_item_times):
        if i not in sim_item_corr:
            miss_item_num += 1
            continue
        for j, wij in sorted(sim_item_corr[i].items(), key=lambda x: x[1], reverse=True)[0:top_k]:
            if j not in interacted_items:
                rank.setdefault(j, 0)

                content_weight = 1.0
                if item_content_sim_dict.get(i, {}).get(j, None) is not None:
                    content_weight += item_content_sim_dict[i][j]
                if item_content_sim_dict.get(j, {}).get(i, None) is not None:
                    content_weight += item_content_sim_dict[j][i]

                time_weight = np.exp(alpha * (time - min_time))
                loc_weight = (0.9 ** (len(interacted_item_times) - loc))
                rank[j] += loc_weight * time_weight * content_weight * wij
    if miss_item_num > 10:
        print('user_id={}, miss_item_num={}'.format(user_id, miss_item_num))

    if item_cnt_dict is not None:
        for loc, item in enumerate(rank):
            rank[item] = re_rank(rank[item], item, user_id, item_cnt_dict, user_cnt_dict, adjust_type=adjust_type)

    sorted_rank_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)

    return sorted_rank_items[0:item_num]


def user_based_recommend(sim_user_corr, user_item_time_dict, user_id, top_k, item_num, alpha=15000,
                         item_cnt_dict=None, user_cnt_dict=None, adjust_type='xtf_v6'):
    item_content_sim_dict = get_glv('item_content_sim_dict')

    rank = {}
    interacted_items = set([i for i, t in user_item_time_dict[user_id]])
    interacted_item_time_list = user_item_time_dict[user_id]
    interacted_num = len(interacted_items)

    min_time = min([t for i, t in interacted_item_time_list])
    time_weight_dict = {i: np.exp(alpha * (t - min_time)) for i, t in interacted_item_time_list}
    loc_weight_dict = {i: 0.9 ** (interacted_num - loc) for loc, (i, t) in enumerate(interacted_item_time_list)}

    for sim_v, wuv in sorted(sim_user_corr[user_id].items(), key=lambda x: x[1], reverse=True)[0:top_k]:
        if sim_v not in user_item_time_dict:
            continue
        for j, j_time in user_item_time_dict[sim_v]:
            if j not in interacted_items:
                rank.setdefault(j, 0)

                content_weight = 1.0
                for loc, (i, t) in enumerate(interacted_item_time_list):
                    loc_weight = loc_weight_dict[i]
                    time_weight = time_weight_dict[i]
                    if item_content_sim_dict.get(i, {}).get(j, None) is not None:
                        content_weight += time_weight * loc_weight * item_content_sim_dict[i][j]

                # weight = np.exp(-15000*abs(j_time-q_time))
                rank[j] += content_weight * wuv

    if item_cnt_dict is not None:
        for loc, item in enumerate(rank):
            rank[item] = re_rank(rank[item], item, user_id, item_cnt_dict, user_cnt_dict, adjust_type=adjust_type)

    rec_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)

    return rec_items[:item_num]


def re_rank(sim, i, u, item_cnt_dict, user_cnt_dict, adjust_type='xtf_v6'):
    '''
    re_rank based on the popularity and similarity
    :param sim:
    :param i:
    :param item_cnt_dict:
    :param adjust_type:
    :return:
    '''
    if adjust_type is None:
        return sim
    elif adjust_type == 'xtf_v6':
        # Log，Linear, 3/4
        if item_cnt_dict.get(i, 1.0) < 4:
            heat = np.log(item_cnt_dict.get(i, 1.0) + 2)
        elif item_cnt_dict.get(i, 1.0) >= 4 and item_cnt_dict.get(i, 1.0) < 10:
            heat = item_cnt_dict.get(i, 1.0)
        else:
            heat = item_cnt_dict.get(i, 1.0) ** 0.75 + 5.0  # 3/4
        sim *= 2.0 / heat

    elif adjust_type == 'zjy_v1':
        user_cnt = user_cnt_dict.get(u, 1.0)

        if item_cnt_dict.get(i, 1.0) < 4:
            heat = np.log(item_cnt_dict.get(i, 1.0) + 2)
        elif item_cnt_dict.get(i, 1.0) >= 4 and item_cnt_dict.get(i, 1.0) < 10:
            if user_cnt > 50:
                heat = item_cnt_dict.get(i, 1.0) * 1
            elif user_cnt > 25:
                heat = item_cnt_dict.get(i, 1.0) * 1.2
            else:
                heat = item_cnt_dict.get(i, 1.0) * 1.6
        else:
            # heat = item_cnt_dict.get(i, 1.0) ** 0.75 + 5.0 # 3/4
            if user_cnt > 50:
                user_cnt_k = 0.4
            elif user_cnt > 10:
                user_cnt_k = 0.1
            else:
                user_cnt_k = 0
            heat = item_cnt_dict.get(i, 1.0) ** user_cnt_k + 10 - 10 ** user_cnt_k  # 3/4
        sim *= 2.0 / heat

    else:
        sim += 2.0 / item_cnt_dict.get(i, 1.0)

    return sim


# fill user to 50 items
def get_predict(df, pred_col, top_fill):
    top_fill = [int(t) for t in top_fill.split(',')]
    scores = [-1 * i for i in range(1, len(top_fill) + 1)]
    ids = list(df['user_id'].unique())
    fill_df = pd.DataFrame(ids * len(top_fill), columns=['user_id'])
    fill_df.sort_values('user_id', inplace=True)
    fill_df['item_id'] = top_fill * len(ids)
    fill_df[pred_col] = scores * len(ids)
    print(len(fill_df))
    df = df.append(fill_df)
    df.sort_values(pred_col, ascending=False, inplace=True)
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    df['rank'] = df.groupby('user_id')[pred_col].rank(method='first', ascending=False)
    df = df[df['rank'] <= 50]
    df = df.groupby('user_id')['item_id'].apply(lambda x: ','.join([str(i) for i in x])).str.split(',', expand=True).reset_index()
    return df

