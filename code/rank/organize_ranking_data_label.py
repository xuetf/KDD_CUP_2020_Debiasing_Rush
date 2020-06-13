from ..conf import *
import pandas as pd


def organize_label_interact_feat_df(click_last_df, click_last_recall_recom_df, phase, is_consider_cold_start=True):
    dfm_df = pd.merge(click_last_recall_recom_df, click_last_df[['user_id', 'item_id', 'time']],
                      on=['user_id', 'item_id'], how='left')
    dfm_df['label'] = dfm_df['time'].apply(lambda x: 0.0 if np.isnan(x) else 1.0)  # time非空代表该click-item被召回了
    del dfm_df['time']

    # merge_time
    click_last_df['day_id'], click_last_df['hour_id'], click_last_df['minute_id'] = zip(
        *click_last_df['time'].apply(time_info))
    dfm_df = pd.merge(dfm_df, click_last_df[['user_id', 'time', 'day_id', 'hour_id', 'minute_id']], on='user_id',
                      how='left')

    # click_last_df里头有些用户的点击没有召回到，即：全部为负样本，导致下采样时，这些用户的负样本可能全被下采样掉了，导致这些用户id丢失；
    # item同理。用户真实点击的item可能没有召回到。
    dfm_df = downsample_by_user(dfm_df)
    dfm_df = dfm_df[use_columns]

    # cold_start_item直接泄露, 这些item可能在infer阶段被recall到，导致item_id缺失
    cold_start_items = set(click_last_df['item_id'].unique()) - set(dfm_df['item_id'].unique())
    if is_consider_cold_start and len(cold_start_items) > 0:
        click_last_cold_start_df = click_last_df[click_last_df['item_id'].isin(cold_start_items)]
        click_last_cold_start_df['label'] = 1.0
        click_last_cold_start_df['phase'] = phase
        for sim_col in sim_columns:
            mean_value = dfm_df[dfm_df['label'] == 1.0][sim_col].mean()
            print('sim_col={}, mean_value={}'.format(sim_col, mean_value))
            click_last_cold_start_df[sim_col] = mean_value
        click_last_cold_start_df = pd.merge(click_last_cold_start_df, dfm_df[['user_id', ] + hist_columns],
                                            on='user_id', how='left')

        print('cold_start_item_num={}, hit_last_cold_start_df_num={}'.format(len(cold_start_items),
                                                                             len(click_last_cold_start_df)))
        dfm_df = dfm_df.append(click_last_cold_start_df[use_columns])

    #     dfm_df = sim_process(dfm_df) # TODO, 移动到召回里头呢？
    return dfm_df


def downsample_by_user(df):
    '''
    percent:多数类别下采样的数量相对于少数类别样本数量的比例
    '''
    data_pos = df[df['label'] != 0]
    data_neg = df[df['label'] == 0]

    def group_neg_sample_func(group_df):
        total_neg_num = len(group_df)
        sample_num = max(int(total_neg_num * 0.002), 1)  # 有些用户召回的数量不足, 取1个
        sample_num = min(sample_num, 5)
        return group_df.sample(n=sample_num, replace=True)

    data_u_neg = data_neg.groupby('user_id', group_keys=False).apply(group_neg_sample_func)  # # 保证user全覆盖
    data_i_neg = data_neg.groupby('item_id', group_keys=False).apply(group_neg_sample_func)  # 保证item全覆盖
    data_neg = data_u_neg.append(data_i_neg)
    data_neg = data_neg.sort_values(['user_id', 'sim']).drop_duplicates(['user_id', 'item_id'], keep='last')

    data = pd.concat([data_neg, data_pos], ignore_index=True)
    data = data.sample(frac=1.0)
    return data

