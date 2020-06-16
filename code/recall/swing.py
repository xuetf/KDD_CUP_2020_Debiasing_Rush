"""
Author:
    Taofeng Xue, xuetfchn@foxmail.com
Reference:
    http://xtf615.com/2018/05/03/recommender-system-survey/
"""

from ..process.recommend_process import *


def swing(df, user_col='user_id', item_col='item_id', time_col='time'):
    # 1. item, (u1,t1), (u2, t2).....
    item_user_df = df.sort_values(by=[item_col, time_col])
    item_user_df = item_user_df.groupby(item_col).apply(
        lambda group: make_user_time_tuple(group, user_col, item_col, time_col)).reset_index().rename(
        columns={0: 'user_id_time_list'})
    item_user_time_dict = dict(zip(item_user_df[item_col], item_user_df['user_id_time_list']))

    user_item_time_dict = defaultdict(list)
    # 2. ((u1, u2), i1, d12)
    u_u_cnt = defaultdict(list)
    item_cnt = defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for u, u_time in user_time_list:
            # just record
            item_cnt[item] += 1
            user_item_time_dict[u].append((item, u_time))

            for relate_u, relate_u_time in user_time_list:
                if relate_u == u:
                    continue

                key = (u, relate_u) if u <= relate_u else (relate_u, u)
                u_u_cnt[key].append((item, np.abs(u_time - relate_u_time)))

    # 3. (i1,i2), sim
    sim_item = {}
    alpha = 5.0
    for u_u, co_item_times in u_u_cnt.items():
        num_co_items = len(co_item_times)
        for i, i_time_diff in co_item_times:
            sim_item.setdefault(i, {})
            for j, j_time_diff in co_item_times:
                if j == i:
                    continue
                weight = 1.0  # np.exp(-15000*(i_time_diff + j_time_diff))
                sim_item[i][j] = sim_item[i].setdefault(j, 0.) + weight / (alpha + num_co_items)
    # 4. norm by item count
    sim_item_corr = sim_item.copy()
    for i, related_items in sim_item.items():
        for j, cij in related_items.items():
            sim_item_corr[i][j] = cij / math.sqrt(item_cnt[i] * item_cnt[j])

    return sim_item_corr, user_item_time_dict
