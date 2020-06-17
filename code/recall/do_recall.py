from . import *


# 基于计算的相似性汇总
def norm_recall_item_score_list(sorted_recall_item_list):
    if len(sorted_recall_item_list) == 0: return sorted_recall_item_list

    assert sorted_recall_item_list[0][1] >= sorted_recall_item_list[-1][1]  # 稍微check下是否排序的
    max_sim = sorted_recall_item_list[0][1]
    min_sim = sorted_recall_item_list[-1][1]

    norm_sorted_recall_item_list = []
    for item, score in sorted_recall_item_list:
        if max_sim > 0:
            norm_score = 1.0 * (score - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 1.0
        else:
            norm_score = 0.0  # topk-fill set to 0.0
        norm_sorted_recall_item_list.append((item, norm_score))
    return norm_sorted_recall_item_list


def norm_user_recall_item_dict(recall_item_dict):
    norm_recall_item_dict = {}
    for u, sorted_recall_item_list in recall_item_dict.items():
        norm_recall_item_dict[u] = norm_recall_item_score_list(sorted_recall_item_list)
    return norm_recall_item_dict


def get_recall_results(item_sim_dict, user_item_dict, target_user_ids=None, item_based=True,
                       item_cnt_dict=None, user_cnt_dict=None, adjust_type='v2'):
    if target_user_ids is None:
        target_user_ids = user_item_dict.keys()
    recall_item_dict = {}

    top50_click_np, top50_click = obtain_top_k_click()

    print('adjust_type={}'.format(adjust_type))

    for u in tqdm(target_user_ids):
        if item_based:
            recall_items = item_based_recommend(item_sim_dict, user_item_dict, u, recommend_num, topk_num,
                                                item_cnt_dict=item_cnt_dict, user_cnt_dict=user_cnt_dict,
                                                adjust_type=adjust_type)
        else:
            recall_items = user_based_recommend(item_sim_dict, user_item_dict, u, recommend_num, topk_num,
                                                item_cnt_dict=item_cnt_dict, user_cnt_dict=user_cnt_dict,
                                                adjust_type=adjust_type)

        if len(recall_items) == 0:
            recall_items = [(top50_click_np[0], 0.0)]  # 防止该用户丢失

        recall_item_dict[u] = recall_items

    return recall_item_dict


# item-cf
# bi-graph
# user-cf
# item-cf
def agg_recall_results(recall_item_dict_list_dict, is_norm=True, ret_type='tuple',
                       weight_dict={}):
    print('aggregate recall results begin....')
    agg_recall_item_dict = {}
    for name, recall_item_dict in recall_item_dict_list_dict.items():
        if is_norm:
            recall_item_dict = norm_user_recall_item_dict(recall_item_dict)
        weight = weight_dict.get(name, 1.0)
        print('name={}, weight={}'.format(name, weight))
        for u, recall_items in recall_item_dict.items():
            agg_recall_item_dict.setdefault(u, {})
            for i, score in recall_items:
                agg_recall_item_dict[u].setdefault(i, 0.0)
                agg_recall_item_dict[u][i] += weight * score  # 累加

    if ret_type == 'tuple':
        agg_recall_item_tuple_dict = {}
        for u, recall_item_dict in agg_recall_item_dict.items():
            sorted_recall_item_tuples = sorted(recall_item_dict.items(), key=lambda x: x[1], reverse=True)
            agg_recall_item_tuple_dict[u] = sorted_recall_item_tuples
        return agg_recall_item_tuple_dict

    if ret_type == 'df':
        recall_u_i_score_pair_list = []
        for u, recall_item_dict in agg_recall_item_dict.items():
            for i, score in recall_item_dict.items():
                recall_u_i_score_pair_list.append((u, i, score))
        recall_df = pd.DataFrame.from_records(recall_u_i_score_pair_list, columns=['user_id', 'item_id', 'sim'])
        return recall_df

    return agg_recall_item_dict


def get_multi_source_sim_dict_results(history_df, recall_methods={'item-cf', 'bi-graph', 'user-cf', 'swing'}):
    recall_sim_pair_dict = {}
    if 'item-cf' in recall_methods:
        print('item-cf item-sim begin')
        item_sim_dict, _ = get_time_dir_aware_sim_item(history_df)
        recall_sim_pair_dict['item-cf'] = item_sim_dict
        print('item-cf item-sim-pair done, pair_num={}'.format(len(item_sim_dict)))

    if 'bi-graph' in recall_methods:
        print('bi-graph item-sim begin')
        item_sim_dict, _ = get_bi_sim_item(history_df)
        recall_sim_pair_dict['bi-graph'] = item_sim_dict
        print('bi-graph item-sim-pair done, pair_num={}'.format(len(item_sim_dict)))

    if 'swing' in recall_methods:
        print('swing item-sim begin')
        item_sim_dict, _ = swing(history_df)
        recall_sim_pair_dict['swing'] = item_sim_dict
        print('swing item-sim-pair done, pair_num={}'.format(len(item_sim_dict)))

    if 'user-cf' in recall_methods:
        print('user-cf user-sim begin')
        user_sim_dict, _ = get_sim_user(history_df)
        recall_sim_pair_dict['user-cf'] = user_sim_dict
        print('user-cf user-sim-pair done, pair_num={}'.format(len(user_sim_dict)))

    return recall_sim_pair_dict


def do_multi_recall_results(recall_sim_pair_dict, user_item_time_dict,
                            target_user_ids=None, ret_type='df', phase=None,
                            item_cnt_dict=None, user_cnt_dict=None,
                            adjust_type='v2', recall_methods={'item-cf', 'bi-graph', 'user-cf', 'swing', 'sr-gnn'}):
    if target_user_ids is None:
        target_user_ids = user_item_time_dict.keys()

    recall_item_list_dict = {}
    for name, sim_dict in recall_sim_pair_dict.items():
        # item-based
        if name in {'item-cf', 'bi-graph', 'swing'}:
            recall_item_dict = get_recall_results(sim_dict, user_item_time_dict, target_user_ids, item_based=True,
                                                  item_cnt_dict=item_cnt_dict, user_cnt_dict=user_cnt_dict,
                                                  adjust_type=adjust_type)
        else:
            recall_item_dict = get_recall_results(sim_dict, user_item_time_dict, target_user_ids, item_based=False,
                                                  item_cnt_dict=item_cnt_dict, user_cnt_dict=user_cnt_dict,
                                                  adjust_type=adjust_type)

        print('{} recall done, recall_user_num={}.'.format(name, len(recall_item_dict)))
        recall_item_list_dict[name] = recall_item_dict

    if 'sr-gnn' in recall_methods:
        standard_sr_gnn_recall_item_dict = read_sr_gnn_results(phase, prefix='standard',
                                                               adjust_type=adjust_type)
        print('read standard sr-gnn results done....')
        pos_weight_sr_gnn_recall_item_dict = read_sr_gnn_results(phase, prefix='pos_node_weight',
                                                                 adjust_type=adjust_type)
        print('read pos_weight sr-gnn results done....')

        recall_item_list_dict['sr_gnn_feat_init_v1'] = standard_sr_gnn_recall_item_dict
        recall_item_list_dict['sr_gnn_pos_weight_v2'] = pos_weight_sr_gnn_recall_item_dict

    return agg_recall_results(recall_item_list_dict, is_norm=True, ret_type=ret_type)


