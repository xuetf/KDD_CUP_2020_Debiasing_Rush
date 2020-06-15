# create offline val data
import pandas as pd
import numpy as np
from ..conf import *

sample_user_num = 1600
if not os.path.exists(offline_answer_path): os.makedirs(offline_answer_path)
if not os.path.exists(offline_test_path): os.makedirs(offline_test_path)
if not os.path.exists(offline_train_path): os.makedirs(offline_train_path)
np.random.seed(1234)  # reproduce-offline


def tr_val_split():
    for phase in range(now_phase + 1):
        click_train = pd.read_csv('{}/{}-{}.csv'.format(online_train_path, train_file_prefix, phase), header=None,
                                  names=['user_id', 'item_id', 'time'])
        all_user_ids = click_train['user_id'].unique()

        sample_user_ids = np.random.choice(all_user_ids, size=sample_user_num, replace=False)

        click_test = click_train[click_train['user_id'].isin(sample_user_ids)]
        click_train = click_train[~click_train['user_id'].isin(sample_user_ids)]

        click_test = click_test.sort_values(by=['user_id', 'time'])
        click_answer = click_test.groupby('user_id').tail(1)
        click_test = click_test.groupby('user_id').apply(lambda x: x[:-1]).reset_index(drop=True)
        click_answer = click_answer[click_answer['user_id'].isin(click_test['user_id'].unique())]  # 防止有些用户只有1个点击数据，去掉
        click_test = click_test[click_test['user_id'].isin(click_answer['user_id'].unique())]
        click_qtime = click_answer[['user_id', 'time']]

        click_train.to_csv(offline_train_path + '/{}-{}.csv'.format(train_file_prefix, phase), index=False, header=None)
        click_answer.to_csv(offline_answer_path + '/{}-{}.csv'.format(infer_answer_file_prefix, phase), index=False,
                            header=None)

        phase_test_path = "{}/{}-{}".format(offline_test_path, test_file_prefix, phase)
        if not os.path.exists(phase_test_path): os.makedirs(phase_test_path)
        click_test.to_csv(phase_test_path + '/{}-{}.csv'.format(test_file_prefix, phase), index=False, header=None)
        click_qtime.to_csv(phase_test_path + '/{}-{}.csv'.format(infer_file_prefix, phase), index=False, header=None)