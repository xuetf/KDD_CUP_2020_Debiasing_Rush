from ..conf import *
import pandas as pd

global item_content_sim_dict


def obtain_user_hist_feat(u, user_item_dict):
    user_hist_seq = [i for i, t in user_item_dict[u]]
    user_hist_time_seq = [t for i, t in user_item_dict[u]]
    user_hist_day_seq, user_hist_hour_seq, user_hist_min_seq = zip(*[time_info(t) for i, t in user_item_dict[u]])
    return [user_hist_seq, user_hist_time_seq, list(user_hist_day_seq), list(user_hist_hour_seq),
            list(user_hist_min_seq)]


def organize_recall_feat_each_user(u, recall_items, user_item_dict, strategy_item_sim_dict, phase):
    user_hist_info = obtain_user_hist_feat(u, user_item_dict)

    # hist-item similarity with recall items
    hist_num = 3
    recall_items_sum_cf_sim2_hist = []
    recall_items_max_cf_sim2_hist = []
    recall_items_cnt_sim2_hist = []

    user_hist_items = user_item_dict[u][::-1][-hist_num:]

    for recall_i, rating in recall_items:
        if rating > 0:
            max_cf_sim2_hist = []
            sum_cf_sim2_hist = []
            cnt_sim2_hist = []
            for hist_i, t in user_hist_items:
                sum_sim_value = 0.0
                max_sim_value = 0.0

                for strategy, item_sim_dict in strategy_item_sim_dict.items():
                    strategy_sim_value = item_sim_dict.get(hist_i, {}).get(recall_i, 0.0) + item_sim_dict.get(recall_i,
                                                                                                              {}).get(
                        hist_i, 0.0)
                    sum_sim_value += strategy_sim_value
                    max_sim_value = max(max_sim_value, strategy_sim_value)

                cnt_sim_value = item_content_sim_dict.get(hist_i, {}).get(recall_i, 0.0) + item_content_sim_dict.get(
                    recall_i, {}).get(hist_i, 0.0)

                sum_cf_sim2_hist.append(sum_sim_value)
                max_cf_sim2_hist.append(max_sim_value)
                cnt_sim2_hist.append(cnt_sim_value)

            while len(sum_cf_sim2_hist) < hist_num:
                sum_cf_sim2_hist.append(0.0)
                max_cf_sim2_hist.append(0.0)
                cnt_sim2_hist.append(0.0)

        else:
            sum_cf_sim2_hist = [0.0] * hist_num
            max_cf_sim2_hist = [0.0] * hist_num
            cnt_sim2_hist = [0.0] * hist_num

        recall_items_sum_cf_sim2_hist.append(sum_cf_sim2_hist)
        recall_items_max_cf_sim2_hist.append(max_cf_sim2_hist)
        recall_items_cnt_sim2_hist.append(cnt_sim2_hist)

    recom_item = []
    for item_rating, sum_cf_sim2_hist, max_cf_sim2_hist, cnt_sim2_hist in zip(recall_items,
                                                                              recall_items_sum_cf_sim2_hist,
                                                                              recall_items_max_cf_sim2_hist,
                                                                              recall_items_cnt_sim2_hist):
        recom_item.append([u, item_rating[0], item_rating[1], phase] + sum_cf_sim2_hist + max_cf_sim2_hist + \
                          cnt_sim2_hist + user_hist_info)

    return recom_item


def organize_recall_feat(recall_item_dict, user_item_hist_dict, item_sim_dict, phase):
    recom_columns = ['user_id', 'item_id', 'sim', 'phase'] + \
                    ['sum_sim2int_1', 'sum_sim2int_2', 'sum_sim2int_3'] + \
                    ['max_sim2int_1', 'max_sim2int_2', 'max_sim2int_3'] + \
                    ['cnt_sim2int_1', 'cnt_sim2int_2', 'cnt_sim2int_3'] + \
                    ['hist_item_id', 'hist_time', 'hist_day_id', 'hist_hour_id', 'hist_minute_id']
    recom_item = []
    for u, recall_items in recall_item_dict.items():
        recom_item.extend(organize_recall_feat_each_user(u, recall_items, user_item_hist_dict, item_sim_dict, phase))

    recall_recom_df = pd.DataFrame(recom_item, columns=recom_columns)
    recall_recom_df['sim_rank_score'] = recall_recom_df.groupby('user_id')['sim'].rank(method='first',
                                                                                       ascending=True) / topk_num

    return recall_recom_df

