import lightgbm as lgb
from ..conf import *


def lgb_main(train_final_df, val_final_df=None):
    print('ranker begin....')
    train_final_df.sort_values(by=['user_id'], inplace=True)
    g_train = train_final_df.groupby(['user_id'], as_index=False).count()["label"].values

    if mode == 'offline':
        val_final_df = val_final_df.sort_values(by=['user_id'])
        g_val = val_final_df.groupby(['user_id'], as_index=False).count()["label"].values

    lgb_ranker = lgb.LGBMRanker(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=300, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.01, min_child_weight=50, random_state=2018,
        n_jobs=-1)  # 300epoch, best, 0.882898, dense_feat  + hist_cnt_sim_feat user_interest_dense_feat

    if mode == 'offline':
        lgb_ranker.fit(train_final_df[lgb_cols], train_final_df['label'], group=g_train,
                       eval_set=[(val_final_df[lgb_cols], val_final_df['label'])], eval_group=[g_val],
                       eval_at=[50], eval_metric=['auc', ],
                       early_stopping_rounds=50, )
    else:
        lgb_ranker.fit(train_final_df[lgb_cols], train_final_df['label'], group=g_train)

    print('train done...')
    return lgb_ranker
