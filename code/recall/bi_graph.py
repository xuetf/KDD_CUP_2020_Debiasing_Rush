"""
Author:
    Taofeng Xue, xuetfchn@foxmail.com

Reference:
    A Simple Recall Method based on Network-based Inference，score:0.18 (phase0-3): https://tianchi.aliyun.com/forum/postDetail?postId=104936
"""

from ..process.recommend_process import *


def get_bi_sim_item(df):
    item_user_time_dict = get_item_user_time_dict(df,)
    user_item_time_dict = get_user_item_time_dict(df)

    item_cnt = defaultdict(int)
    for user, item_times in tqdm(user_item_time_dict.items()):
        for i, t in item_times:
            item_cnt[i] += 1

    sim_item = {}

    for item, user_time_lists in tqdm(item_user_time_dict.items()):

        sim_item.setdefault(item, {})

        for u, item_time in user_time_lists:

            tmp_len = len(user_item_time_dict[u])

            for relate_item, related_time in user_item_time_dict[u]:
                sim_item[item].setdefault(relate_item, 0)
                weight = np.exp(-15000 * np.abs(related_time - item_time))
                sim_item[item][relate_item] += weight / (math.log(len(user_time_lists) + 1) * math.log(tmp_len + 1))

    return sim_item, user_item_time_dict
