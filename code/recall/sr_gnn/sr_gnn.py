from .construct_sr_gnn_train_data import *
import sys
import re
sys.path.append('lib')


def find_checkpoint_path(phase, checkpoint_prefix='session_id'):
    checkpoint_dir = 'tmp/model_saved/{}/{}'.format(mode, phase)
    step_max = 0
    re_cp = re.compile("{}-(\d+)\.".format(checkpoint_prefix))
    for file in os.listdir(checkpoint_dir):
        so = re_cp.search(file)
        if so:
            step = int(so.group(1))
            step_max = step if step > step_max else step_max
    checkpoint_path = '{}/{}-{}'.format(checkpoint_dir, checkpoint_prefix, step_max)
    print('CheckPoint: {}'.format(checkpoint_path))
    return checkpoint_path


if __name__ == '__main__':
    processed_item_feat_df, item_content_vec_dict = obtain_entire_item_feat_df()
    phase_item_cnt_dict = {}  # 7: 45194, 8: 44979, 9: 44365
    for phase in range(start_phase, now_phase+1):
        item_cnt = construct_sr_gnn_train_data(phase, item_content_vec_dict, is_use_whole_click=True)
        phase_item_cnt_dict[phase] = item_cnt
    print('construct train data done...')

    # running model
    for phase in range(start_phase, now_phase+1):
        print('phase={}'.format(phase))
        model_path = 'tmp/model_saved/{}/{}'.format(mode, phase)
        if not os.path.exists(model_path): os.makedirs(model_path)

        file_path = '{}/{}'.format(sr_gnn_root_dir, phase)
        if os.path.exists(model_path):
            print('model_path={} exists, delete'.format(model_path))
            os.system("rm -rf {}".format(model_path))
        item_cnt = phase_item_cnt_dict[phase]
        os.system("python3 main.py --task train --node_count {item_cnt} "
                  "--checkpoint_path {model_path}/session_id --train_input {file_path}/train_item_seq_enhanced.txt "
                  "--test_input {file_path}/test_item_seq.txt --gru_step 2 --epochs 10 "
                  "--lr 0.001 --lr_dc 2 --dc_rate 0.1 --early_stop_epoch 3 --hidden_size 256 --batch_size 256 "
                  "--max_len 20 --has_uid True --feature_init {file_path}/item_embed_mat.npy --sigma 10 "
                  "--sq_max_len 5 --node_weight True  --node_weight_trainable True".format(item_cnt=item_cnt,
                                                                                           model_path=model_path,
                                                                                           file_path=file_path))
        # generate rec
        checkpoint_path = find_checkpoint_path(phase)
        prefix = 'pos_node_weight_'

        rec_path = '{}/{}rec.txt'.format(file_path, prefix)
        os.system("python3 main.py --task recommend --node_count {item_cnt} --checkpoint_path {checkpoint_path} "
                  "--item_lookup {file_path}/item_lookup.txt --recommend_output {rec_path} "
                  "--session_input {file_path}/test_user_sess.txt --gru_step 2 "
                  "--hidden_size 256 --batch_size 256 --rec_extra_count 50 --has_uid True "
                  "--feature_init {file_path}/item_embed_mat.npy "
                  "--max_len 10 --sigma 10 --sq_max_len 5 --node_weight True "
                  "--node_weight_trainable True".format(item_cnt=item_cnt, checkpoint_path=checkpoint_path,
                                                        file_path=file_path, rec_path=rec_path,))