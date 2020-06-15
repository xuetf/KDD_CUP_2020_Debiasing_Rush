import pickle
import pandas as pd
from ..conf import *


def make_item_sim_tuple(group_df):
    group_df = group_df.sort_values(by=['sim'], ascending=False)
    item_score_tuples = list(zip(group_df['item_id'], group_df['sim']))
    return item_score_tuples


def save_recall_df_as_user_tuples_dict(total_recom_df, phase_full_sim_dict, prefix=''):
    save_path = os.path.join(user_data_dir, 'recall', mode)
    if not os.path.exists(save_path): os.makedirs(save_path)

    pickle.dump(total_recom_df, open(os.path.join(save_path, prefix + '_total_recall_df.pkl'), 'wb'))

    for phase in range(start_phase, now_phase + 1):
        phase_df = total_recom_df[total_recom_df['phase'] == phase]
        phase_user_item_score_dict = recall_df2dict(phase_df)
        phase_sim_dict = phase_full_sim_dict[phase]

        pickle.dump(phase_user_item_score_dict,
                    open(os.path.join(save_path, '{}_phase_{}.pkl'.format(prefix, phase)), 'wb'))
        pickle.dump(phase_sim_dict, open(os.path.join(save_path, '{}_phase_{}_sim.pkl'.format(prefix, phase)), 'wb'))


def sub2_df(filename):
    rec_items = []
    constant_sim = 100
    with open(filename) as f:
        for line in f:
            row = line.strip().split(",")
            uid = int(row[0])
            iids = row[1:]
            phase = uid % 11
            for idx, iid in enumerate(iids):
                rec_items.append((uid, int(iid), constant_sim - idx, phase))

    return pd.DataFrame(rec_items, columns=['user_id', 'item_id', 'sim', 'phase'])


def recall_df2dict(phase_df):
    phase_df = phase_df.groupby('user_id').apply(make_item_sim_tuple).reset_index().rename(
        columns={0: 'item_score_list'})
    item_score_list = phase_df['item_score_list'].apply(
        lambda item_score_list: sorted(item_score_list, key=lambda x: x[1], reverse=True))
    phase_user_item_score_dict = dict(zip(phase_df['user_id'], item_score_list))
    return phase_user_item_score_dict


def recall_dict2df(recall_item_score_dict):
    recom_list = []
    for u, item_score_list in recall_item_score_dict.items():
        for item, score in item_score_list:
            recom_list.append((u, item, score))
    return pd.DataFrame(recom_list, columns=['user_id', 'item_id', 'sim'])

