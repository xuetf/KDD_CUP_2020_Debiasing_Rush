from ...process.recommend_process import *
from ...process.load_data import *
from ...process.convert_data import *


def filter_df(recom_df, phase, is_item_cnt_weight=False, adjust_type='xtf_v6'):
    print(len(recom_df))
    filter_num = 0

    all_click, click_q_time = get_phase_click(phase)
    phase_whole_click = get_whole_phase_click(all_click, click_q_time)

    if mode == 'online':
        user_item_hist_dict = get_user_item_dict(phase_whole_click)
    else:
        user_item_hist_dict = get_user_item_dict(all_click)

    item_cnt_dict = all_click.groupby('item_id')['user_id'].count().to_dict()
    user_cnt_dict = all_click.groupby('user_id')['item_id'].count().to_dict()

    recom_list = []
    for row in recom_df.itertuples(index=False):
        uid = int(row.user_id)
        iid = int(row.item_id)
        if uid in user_item_hist_dict and iid in user_item_hist_dict[uid]:
            filter_num += 1
            continue
        sim = row.sim
        if is_item_cnt_weight:
            sim = re_rank(row.sim, iid, uid, item_cnt_dict, user_cnt_dict, adjust_type=adjust_type)
        #             sim = row.sim * 2.0 / item_cnt_dict.get(iid, 1.0)
        recom_list.append((uid, iid, sim, row.phase))

    print('num={}, filter_num={}'.format(len(recom_list), filter_num))
    filter_recom_df = pd.DataFrame(recom_list, columns=['user_id', 'item_id', 'sim', 'phase'])
    return filter_recom_df


def read_sr_gnn_results(phase, prefix='standard', adjust_type='xtf_v6'):
    print('sr-gnn begin...')
    sr_gnn_rec_path = '{}/{}/{}_rec.txt'.format(sr_gnn_root_dir, phase, prefix)  # standard_rec.txt + pos_node_weight_rec.txt
    print('path={}'.format(sr_gnn_rec_path))
    rec_user_item_dict = {}
    with open(sr_gnn_rec_path) as f:
        for line in f:
            try:
                row = eval(line)
                uid = row[0]
                iids = row[1]
                iids = [(int(iid), float(score)) for iid, score in iids]
                iids = sorted(iids, key=lambda x: x[1], reverse=True)
                rec_user_item_dict[int(uid)] = iids
            except Exception as e:
                print(e)
                exit(-1)
    print('read sr-gnn done, num={}'.format(rec_user_item_dict))
    recom_df = recall_dict2df(rec_user_item_dict)
    recom_df['phase'] = phase
    recom_df = filter_df(recom_df, phase, is_item_cnt_weight=True, adjust_type=adjust_type)
    recall_user_item_score_dict = recall_df2dict(recom_df)
    return recall_user_item_score_dict
