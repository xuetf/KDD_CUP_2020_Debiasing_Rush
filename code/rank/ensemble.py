import pickle
from ..process.load_data import *
from ..process.recommend_process import *


def norm_sim(sim_df, weight=0.0):
    # print(sim_df.head())
    min_sim = sim_df.min()
    max_sim = sim_df.max()
    if max_sim == min_sim:
        sim_df = sim_df.apply(lambda sim: 1.0)
    else:
        sim_df = sim_df.apply(lambda sim: 1.0 * (sim - min_sim) / (max_sim - min_sim))

    sim_df = sim_df.apply(lambda sim: sim + weight)  # plus one
    return sim_df


def ensemble(output_ranking_filename):
    # ensemble lgb+din
    lgb_output_file = 'ranker-' + output_ranking_filename + '-pkl'
    # read lgb
    lgb_ranker_df = pickle.load(
        open('{}/{}'.format(output_path, lgb_output_file), 'rb'))
    lgb_ranker_df['sim'] = lgb_ranker_df.groupby('user_id')['sim'].transform(lambda df: norm_sim(df))

    # read din
    din_output_file = 'din-' + output_ranking_filename + '-pkl'
    din_df = pickle.load(
        open('{}/{}'.format(output_path, din_output_file), 'rb'))
    din_df['sim'] = din_df.groupby('user_id')['sim'].transform(lambda df: norm_sim(df))

    # fuse lgb and din
    din_lgb_full_df = lgb_ranker_df.append(din_df)
    din_lgb_full_df = din_lgb_full_df.groupby(['user_id', 'item_id', 'phase'])['sim'].sum().reset_index()

    online_top50_click_np, online_top50_click = obtain_top_k_click()
    res3 = get_predict(din_lgb_full_df, 'sim', online_top50_click)
    res3.to_csv(output_path + '/result.csv', index=False, header=None)


