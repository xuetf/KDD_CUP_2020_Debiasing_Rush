"""
Author:
    Taofeng Xue, xuetfchn@foxmail.com, xtf615.com
Reference:
    http://xtf615.com/2018/05/03/recommender-system-survey/
"""

from ..process.recommend_process import *


# user-cf
def get_sim_user(df):
    # user_min_time_dict = get_user_min_time_dict(df, user_col, item_col, time_col) # user first time
    # history
    user_item_time_dict = get_user_item_time_dict(df)
    # item, [u1, u2, ...,]
    item_user_time_dict = get_item_user_time_dict(df)

    sim_user = {}
    user_cnt = defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        num_users = len(user_time_list)
        for u, t in user_time_list:
            user_cnt[u] += 1
            sim_user.setdefault(u, {})
            for relate_user, relate_t in user_time_list:
                # time_diff_relate_u = 1.0/(1.0+10000*abs(relate_t-t))
                if u == relate_user:
                    continue
                sim_user[u].setdefault(relate_user, 0)
                weight = 1.0
                sim_user[u][relate_user] += weight / math.log(1 + num_users)

    sim_user_corr = sim_user.copy()
    for u, related_users in tqdm(sim_user.items()):
        for v, cuv in related_users.items():
            sim_user_corr[u][v] = cuv / math.sqrt(user_cnt[u] * user_cnt[v])

    return sim_user_corr, user_item_time_dict

