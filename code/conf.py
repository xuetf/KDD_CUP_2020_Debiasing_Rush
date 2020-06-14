import os
import time

data_dir = 'data'
user_data_dir = 'user_data'

# online prediction data
online_train_path = os.path.join(data_dir, 'underexpose_train')
online_test_path = os.path.join(data_dir, 'underexpose_test')

# offline evaluation data
offline_train_path = os.path.join(user_data_dir, 'offline_underexpose_train')
offline_test_path = os.path.join(user_data_dir, 'offline_underexpose_test')
offline_answer_path = os.path.join(user_data_dir, 'offline_underexpose_answer')

train_file_prefix = 'underexpose_train_click'
test_file_prefix = 'underexpose_test_click'
infer_file_prefix = 'underexpose_test_qtime'
infer_answer_file_prefix = 'underexpose_test_qtime_with_answer'

item_feat_file_path = os.path.join(online_train_path, 'underexpose_item_feat.csv')
user_feat_file_path = os.path.join(online_train_path, 'underexpose_user_feat.csv')


# global variables to control online or offline

mode = 'online'
now_phase = 9
start_phase = 7
train_path = online_train_path if mode == 'online' else offline_train_path
test_path = online_test_path if mode == 'online' else offline_test_path


online_output_path = 'prediction_result'
offline_output_path = os.path.join(user_data_dir, 'prediction_result')

output_path = online_output_path if mode == 'online' else offline_output_path
if not os.path.exists(output_path): os.mkdir(output_path)

recommend_num = 800  # iterate number
topk_num = 200  # final recall number of each method

sr_gnn_root_dir = os.path.join(user_data_dir,'sr-gnn', mode)
if not os.path.exists(sr_gnn_root_dir): os.mkdir(sr_gnn_root_dir)

# ranking
w2v_dim = 32
basic_columns = ['user_id', 'item_id', 'phase', 'label', ]
time_columns = ['time', 'day_id', 'hour_id', 'minute_id']
hist_columns = ['hist_item_id', 'hist_time', 'hist_day_id', 'hist_hour_id', 'hist_minute_id', ]
sim_columns = ['sim', 'sum_sim2int_1', 'sum_sim2int_2', 'sum_sim2int_3'] + \
              ['max_sim2int_1', 'max_sim2int_2', 'max_sim2int_3', 'sim_rank_score'] + \
              ['cnt_sim2int_1', 'cnt_sim2int_2', 'cnt_sim2int_3']
use_columns = basic_columns + hist_columns + sim_columns + time_columns


max_seq_len = 10
time_feat = ['day_id', 'hour_id']  # , 'minute_id']  # no need to sparse encoder

sparse_feat = ['user_id', 'item_id', ] + time_feat  # + user_sparse_feat

# sim_dense_feat =  ['sim', 'exp_sim', 'sim2int_1', 'sim2int_2', 'sim2int_3'] + ['cnt_sim2int_1', 'cnt_sim2int_2', 'cnt_sim2int_3'] # , 'sim_rank_score']
sim_dense_feat = ['sim', 'sum_sim2int_1', 'sum_sim2int_2', 'sum_sim2int_3'] + \
                 ['max_sim2int_1', 'max_sim2int_2', 'max_sim2int_3', 'sim_rank_score'] + \
                 ['cnt_sim2int_1', 'cnt_sim2int_2', 'cnt_sim2int_3']

hist_cnt_sim_feat = ['txt_cnt_sim_last_1', 'img_cnt_sim_last_1', 'cnt_sim_last_1'] + \
                    ['txt_cnt_sim_last_2', 'img_cnt_sim_last_2', 'cnt_sim_last_2'] + \
                    ['txt_cnt_sim_last_3', 'img_cnt_sim_last_3', 'cnt_sim_last_3']

hist_time_diff_feat = ['time_diff_1', 'time_diff_2', 'time_diff_3']

w2v_sim_feat = ['w2v_sim_last_1', 'w2v_sim_last_2', 'w2v_sim_last_3']

user_w2v_embed_feat = ['user_w2v_embed_{}'.format(i) for i in range(128)]
item_w2v_embed_feat = ['item_w2v_embed_{}'.format(i) for i in range(128)]
w2v_user_item_feat = ['w2v_sim'] + user_w2v_embed_feat + item_w2v_embed_feat


t = (2020, 4, 10, 0, 0, 0, 0, 0, 0)
time_end = time.mktime(t)
max_day, max_hour, max_miniute = 7, 24, 60
def time_info(time_delta):
    import time
    timestamp = time_end * time_delta
    struct_time = time.gmtime(timestamp)
    day, hour, mini = struct_time.tm_wday + 1, struct_time.tm_hour + 1, struct_time.tm_min + 1
    return day, hour, mini


txt_dense_feat = ['txt_embed_' + str(i) for i in range(128)]
img_dense_feat = ['img_embed_' + str(i) for i in range(128)]
item_dense_feat = txt_dense_feat + img_dense_feat
dense_feat = item_dense_feat + sim_dense_feat  # + item_statistic_feat
var_len_feat = ['hist_item_id'] + ['hist_{}'.format(feat) for feat in time_feat]
user_interest_dense_feat = ['interest_' + col for col in item_dense_feat] + ['interest_degree', 'txt_interest_degree',
                                                                             'img_interest_degree', ]


lgb_cols = dense_feat + user_interest_dense_feat + hist_cnt_sim_feat + hist_time_diff_feat + w2v_sim_feat
use_feats = ['user_id', 'item_id'] + ['hist_item_id', ] + lgb_cols