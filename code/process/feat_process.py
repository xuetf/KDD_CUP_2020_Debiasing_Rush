from .recommend_process import *
from .load_data import *


def read_item_feat_df():
    print('begin read item df...')
    item_feat_cols = ['item_id', ] + ['txt_embed_' + str(i) for i in range(128)] + ['img_embed_' + str(i) for i in range(128)]
    item_feat_df = pd.read_csv(item_feat_file_path, header=None, names=item_feat_cols)
    item_feat_df['txt_embed_0'] = item_feat_df['txt_embed_0'].apply(lambda x: float(x[1:]))
    item_feat_df['txt_embed_127'] = item_feat_df['txt_embed_127'].apply(lambda x: float(x[:-1]))
    item_feat_df['img_embed_0'] = item_feat_df['img_embed_0'].apply(lambda x: float(x[1:]))
    item_feat_df['img_embed_127'] = item_feat_df['img_embed_127'].apply(lambda x: float(x[:-1]))
    return item_feat_df


def process_item_feat(item_feat_df):
    processed_item_feat_df = item_feat_df.copy()
    # norm
    txt_item_feat_np = processed_item_feat_df[txt_dense_feat].values
    img_item_feat_np = processed_item_feat_df[img_dense_feat].values
    txt_item_feat_np = txt_item_feat_np / np.linalg.norm(txt_item_feat_np, axis=1, keepdims=True)
    img_item_feat_np = img_item_feat_np / np.linalg.norm(img_item_feat_np, axis=1, keepdims=True)
    processed_item_feat_df[txt_dense_feat] = pd.DataFrame(txt_item_feat_np, columns=txt_dense_feat)
    processed_item_feat_df[img_dense_feat] = pd.DataFrame(img_item_feat_np, columns=img_dense_feat)

    # item_feat_dict = dict(zip(processed_item_feat_df['item_id'], processed_item_feat_df[dense_feat].values))
    return processed_item_feat_df, dense_feat


def fill_item_feat(processed_item_feat_df, item_content_vec_dict):
    online_total_click = get_online_whole_click()

    all_click_feat_df = pd.merge(online_total_click, processed_item_feat_df, on='item_id', how='left')
    # 缺失值
    missed_items = all_click_feat_df[all_click_feat_df['txt_embed_0'].isnull()]['item_id'].unique()
    user_item_time_hist_dict = get_user_item_time_dict(online_total_click)

    # co-occurance
    co_occur_dict = {}
    window = 5

    def cal_occ(sentence):
        for i, word in enumerate(sentence):
            hist_len = len(sentence)
            co_occur_dict.setdefault(word, {})
            for j in range(max(i - window, 0), min(i + window, hist_len)):
                if j == i or word == sentence[j]: continue
                loc_weight = (0.9 ** abs(i - j))
                co_occur_dict[word].setdefault(sentence[j], 0)
                co_occur_dict[word][sentence[j]] += loc_weight

    for u, hist_item_times in user_item_time_hist_dict.items():
        hist_items = [i for i, t in hist_item_times]
        cal_occ(hist_items)

    # fill
    miss_item_content_vec_dict = {}
    for miss_item in missed_items:
        co_occur_item_dict = co_occur_dict[miss_item]
        weighted_vec = np.zeros(256)
        sum_weight = 0.0
        for co_item, weight in co_occur_item_dict.items():

            if co_item in item_content_vec_dict:
                sum_weight += weight
                co_item_vec = item_content_vec_dict[co_item]
                weighted_vec += weight * co_item_vec

        weighted_vec /= sum_weight
        txt_item_feat_np = weighted_vec[0:128] / np.linalg.norm(weighted_vec[0:128])
        img_item_feat_np = weighted_vec[128:] / np.linalg.norm(weighted_vec[128:])
        cnt_vec = np.concatenate([txt_item_feat_np, img_item_feat_np])
        miss_item_content_vec_dict[miss_item] = cnt_vec

    miss_item_feat_df = pd.DataFrame()
    miss_item_feat_df[item_dense_feat] = pd.DataFrame(miss_item_content_vec_dict.values(),
                                                      columns=item_dense_feat)
    miss_item_feat_df['item_id'] = list(miss_item_content_vec_dict.keys())
    miss_item_feat_df = miss_item_feat_df[['item_id'] + item_dense_feat]

    return miss_item_feat_df, miss_item_content_vec_dict


def obtain_entire_item_feat_df():
    item_feat_df = read_item_feat_df()
    processed_item_feat_df, _ = process_item_feat(item_feat_df)
    item_content_vec_dict = dict(zip(processed_item_feat_df['item_id'], processed_item_feat_df[item_dense_feat].values))
    miss_item_feat_df, miss_item_content_vec_dict = fill_item_feat(processed_item_feat_df, item_content_vec_dict)
    processed_item_feat_df = processed_item_feat_df.append(miss_item_feat_df)
    processed_item_feat_df = processed_item_feat_df.reset_index(drop=True)
    item_content_vec_dict.update(miss_item_content_vec_dict)
    return processed_item_feat_df, item_content_vec_dict
