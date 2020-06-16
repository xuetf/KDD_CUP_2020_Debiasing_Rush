from ...process.load_data import *
from ...process.recommend_process import *
from ...process.feat_process import *
from sklearn.preprocessing import LabelEncoder
import numpy as np


def construct_sr_gnn_train_data(target_phase, item_content_vec_dict, is_use_whole_click=True):

    # step 1: obtain original training data
    sr_gnn_dir = '{}/{}'.format(sr_gnn_root_dir, target_phase)
    if not os.path.exists(sr_gnn_dir): os.makedirs(sr_gnn_dir)
    all_click, click_q_time = get_phase_click(target_phase)
    phase_click = all_click
    if is_use_whole_click:
        print('using whole click to build training data')
        phase_whole_click = get_whole_phase_click(all_click, click_q_time)
        phase_click = phase_whole_click
    user_item_time_hist_dict = get_user_item_time_dict(phase_click)

    # step 2: encode the iid and uid
    # sparse features one-hot
    lbe = LabelEncoder()
    lbe.fit(phase_click['item_id'].astype(str))
    item_raw_id2_idx_dict = dict(zip(lbe.classes_,
                                     lbe.transform(lbe.classes_) + 1, ))  # 得到字典
    item_cnt = len(item_raw_id2_idx_dict)
    print(item_cnt)

    lbe = LabelEncoder()
    lbe.fit(phase_click['user_id'].astype(str))
    user_raw_id2_idx_dict = dict(zip(lbe.classes_,
                                     lbe.transform(lbe.classes_) + 1, ))  # dictionary
    user_cnt = len(user_raw_id2_idx_dict)
    print(user_cnt)

    # step 3: obtain feat to initialize embedding
    # step 3.1: item embedding
    item_embed_np = np.zeros((item_cnt + 1, 256))
    for raw_id, idx in item_raw_id2_idx_dict.items():
        vec = item_content_vec_dict[int(raw_id)]
        item_embed_np[idx, :] = vec
    np.save(open(sr_gnn_dir + '/item_embed_mat.npy', 'wb'), item_embed_np)
    # step 3.2: obtain user embedding (in fact, we don't use user embedding due to its limited performance)
    user_embed_np = np.zeros((user_cnt + 1, 256))
    for raw_id, idx in user_raw_id2_idx_dict.items():
        hist = user_item_time_hist_dict[int(raw_id)]
        vec = weighted_agg_content(hist, item_content_vec_dict)
        user_embed_np[idx, :] = vec
    np.save(open(sr_gnn_dir + '/user_embed_mat.npy', 'wb'), user_embed_np)

    # step 4: obtain item sequences based on the training data, i.e, train sequences, validate sequences, infer sequences
    full_user_item_dict = get_user_item_time_dict(phase_click)
    print(len(full_user_item_dict))
    # 4.1 train sequences
    train_user_hist_seq_dict = {}
    for u, hist_seq in full_user_item_dict.items():
        if len(hist_seq) > 1:
            train_user_hist_seq_dict[u] = hist_seq
    train_users = train_user_hist_seq_dict.keys()
    print(len(train_user_hist_seq_dict))
    # 4.2 validate sequences and infer sequences
    test_users = click_q_time['user_id'].unique()
    test_user_hist_seq_dict = {}
    infer_user_hist_seq_dict = {}
    for test_u in test_users:
        if test_u not in full_user_item_dict:
            print('test-user={} not in train/test data'.format(test_u))
            continue
        if len(full_user_item_dict[test_u]) > 1:
            test_user_hist_seq_dict[test_u] = full_user_item_dict[test_u]
            if test_u in train_user_hist_seq_dict:
                if len(train_user_hist_seq_dict[test_u][: -1]) > 1:
                    train_user_hist_seq_dict[test_u] = train_user_hist_seq_dict[test_u][: -1]  # last one not train, use just for test
                else:
                    del train_user_hist_seq_dict[test_u]

        infer_user_hist_seq_dict[test_u] = full_user_item_dict[test_u]

    print(len(train_user_hist_seq_dict))
    print(len(test_user_hist_seq_dict))
    print(len(infer_user_hist_seq_dict))

    # step 5: generate the data for the SR-GNN model
    def gen_data(is_attach_user=False):
        with open(sr_gnn_dir + '/train_item_seq.txt', 'w') as f_seq, open(sr_gnn_dir + '/train_user_sess.txt',
                                                                               'w') as f_user:
            for u in train_users:
                u_idx = user_raw_id2_idx_dict[str(u)]
                hist_item_time_seq = train_user_hist_seq_dict[u]
                hist_item_seq = [str(item_raw_id2_idx_dict[str(item)]) for item, time in hist_item_time_seq]
                if is_attach_user:
                    hist_item_seq_sess = [str(u_idx), ] + hist_item_seq
                else:
                    hist_item_seq_sess = hist_item_seq

                hist_item_seq_str = " ".join(hist_item_seq_sess)
                f_seq.write(hist_item_seq_str + '\n')

                # infer
                if is_attach_user:
                    hist_item_user_sess = [str(u), str(u_idx)] + hist_item_seq
                else:
                    hist_item_user_sess = [str(u), ] + hist_item_seq
                hist_item_user_sess_str = " ".join(hist_item_user_sess)
                f_user.write(hist_item_user_sess_str + '\n')

        with open(sr_gnn_dir + '/test_item_seq.txt', 'w') as f_seq, open(sr_gnn_dir + '/test_user_sess.txt',
                                                                              'w') as f_user:
            for u in test_users:
                # test
                if u in test_user_hist_seq_dict:
                    u_idx = user_raw_id2_idx_dict[str(u)]
                    hist_item_time_seq = test_user_hist_seq_dict[u]
                    hist_item_seq = [str(item_raw_id2_idx_dict[str(item)]) for item, time in hist_item_time_seq]

                    if is_attach_user:
                        hist_item_seq_sess = [str(u_idx), ] + hist_item_seq
                    else:
                        hist_item_seq_sess = hist_item_seq

                    hist_item_seq_str = " ".join(hist_item_seq_sess)
                    f_seq.write(hist_item_seq_str + '\n')

                if u in infer_user_hist_seq_dict:
                    hist_item_time_seq = infer_user_hist_seq_dict[u]
                    hist_item_seq = [str(item_raw_id2_idx_dict[str(item)]) for item, time in hist_item_time_seq]

                    if is_attach_user:
                        hist_item_user_sess = [str(u), str(u_idx)] + hist_item_seq
                    else:
                        hist_item_user_sess = [str(u), ] + hist_item_seq

                    hist_item_user_sess_str = " ".join(hist_item_user_sess)
                    f_user.write(hist_item_user_sess_str + '\n')

        with open(sr_gnn_dir + '/item_lookup.txt', 'w') as f_item_map:
            for raw_id, idx in item_raw_id2_idx_dict.items():
                f_item_map.write("{} {}\n".format(idx, raw_id))

    gen_data(is_attach_user=True)

     # step 6: enhance the data for the SR-GNN model to enrich the data. (convert the long sequences to multiple short sequences)
    def enhance_data(is_attach_user=False):
        np.random.seed(1234)
        count = 0
        max_len = 10
        tmp_max = 0
        with open(sr_gnn_dir + '/data/train_item_seq.txt', 'r') as f_in, open(
                sr_gnn_dir + '/data/train_item_seq_enhanced.txt', 'w') as f_out:
            for line in f_in:
                row = line.strip().split()

                if is_attach_user:
                    uid = row[0]
                    iids = row[1:]
                else:
                    iids = row

                end_step_1 = max(2, np.random.poisson(4))
                end_step_2 = len(iids) + 1

                if end_step_2 > end_step_1:
                    for i in range(end_step_1, end_step_2):
                        count += 1
                        start_end = max(i - max_len, 0)
                        tmp_max = max(tmp_max, len(iids[start_end: i]))
                        sampled_seq = iids[start_end: i]

                        if is_attach_user:
                            sampled_seq = [str(uid), ] + sampled_seq

                        f_out.write(' '.join(sampled_seq) + '\n')
                else:
                    count += 1
                    f_out.write(line)
        print("Done, Output Lines: {}".format(count))
        print(tmp_max)

    enhance_data(is_attach_user=True)

    return item_cnt # item_cnt just return as the input args for SR-GNN


def weighted_agg_content(hist_item_id_list, item_content_vec_dict):
    # weighted user behavior sequences to obtain user initial embedding
    weighted_vec = np.zeros(128*2)
    hist_num = len(hist_item_id_list)
    sum_weight = 0.0
    for loc, (i,t) in enumerate(hist_item_id_list):
        loc_weight = (0.9**(hist_num-loc))
        if i in item_content_vec_dict:
            sum_weight += loc_weight
            weighted_vec += loc_weight*item_content_vec_dict[i]
    if sum_weight != 0:
        weighted_vec /= sum_weight
        txt_item_feat_np = weighted_vec[0:128] / np.linalg.norm(weighted_vec[0:128])
        img_item_feat_np = weighted_vec[128:] / np.linalg.norm(weighted_vec[128:])
        weighted_vec = np.concatenate([txt_item_feat_np,  img_item_feat_np])
    else:
        print('zero weight...')
    return weighted_vec