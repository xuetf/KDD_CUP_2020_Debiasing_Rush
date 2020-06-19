from ..recall.do_recall_multi_processing import *
from .organize_ranking_data_recall_feat import *
from .organize_ranking_data_label import *
from ..process.load_data import *
from .organize_ranking_data_info_feat import *
from ..conf import *
from ..global_variables import *


def get_history_and_last_click_df(click_df):
    click_df = click_df.sort_values(by=['user_id', 'time'])
    click_last_df = click_df.groupby('user_id').tail(1)

    # 用户只有1个点击时，history为空了，导致训练的时候这个用户不可见, 此时默认一下该用户泄露
    def hist_func(user_df):
        num = len(user_df)
        if num == 1:
            return user_df
        else:
            return user_df[:-1]

    click_history_df = click_df.groupby('user_id').apply(hist_func).reset_index(drop=True)

    return click_history_df, click_last_df


def sliding_obtain_training_df(c, is_silding_compute_sim=False):
    compute_mode = 'once' if not is_silding_compute_sim else 'multi'
    save_training_path = os.path.join(user_data_dir, 'training', mode, compute_mode, str(c))

    if os.path.exists(os.path.join(save_training_path, 'step_user_recall_item_dict.pkl')):
        print('phase={} already done...'.format(c))
        return
    all_click, click_q_time = get_phase_click(c)

    click_history_df = all_click
    recall_methods = {'item-cf', 'bi-graph', 'user-cf', 'swing'}

    if not os.path.exists(save_training_path): os.makedirs(save_training_path)

    total_step = 10
    step = 0
    full_sim_pair_dict = get_multi_source_sim_dict_results_multi_processing(click_history_df,
                                                                            recall_methods=recall_methods)
    pickle.dump(full_sim_pair_dict, open(os.path.join(save_training_path, 'full_sim_pair_dict.pkl'), 'wb'))

    step_user_recall_item_dict = {}
    step_strategy_sim_pair_dict = {}

    while step < total_step:
        print('step={}'.format(step))
        click_history_df, click_last_df = get_history_and_last_click_df(click_history_df)  # override click_history_df
        user_item_time_dict = get_user_item_time_dict(click_history_df)

        if is_silding_compute_sim:
            sim_pair_dict = get_multi_source_sim_dict_results_multi_processing(click_history_df,
                                                                               recall_methods=recall_methods)  # re-compute
        else:
            sim_pair_dict = full_sim_pair_dict

        user_recall_item_dict = do_multi_recall_results_multi_processing(sim_pair_dict, user_item_time_dict,
                                                                         ret_type='tuple',
                                                                         recall_methods=recall_methods)

        step_user_recall_item_dict[step] = user_recall_item_dict
        if is_silding_compute_sim:
            step_strategy_sim_pair_dict[step] = sim_pair_dict
        # step_user_hist_item_time_dict[step] = user_item_time_dict
        step += 1

    pickle.dump(step_user_recall_item_dict,
                open(os.path.join(save_training_path, 'step_user_recall_item_dict.pkl'), 'wb'))

    if is_silding_compute_sim:
        pickle.dump(step_strategy_sim_pair_dict,
                    open(os.path.join(save_training_path, 'step_strategy_sim_pair_dict.pkl'), 'wb'))

    # validation/test recall results based on full_sim_pair_dict
    # user-cf depend on sim-user history, so use all-click; test user history will not occur in train, so it's ok
    print('obtain validate/test recall data')
    if mode == 'offline':
        all_user_item_dict = get_user_item_time_dict(all_click)

        val_user_recall_item_dict = do_multi_recall_results_multi_processing(full_sim_pair_dict,
                                                                             all_user_item_dict,
                                                                             target_user_ids=click_q_time['user_id'].unique(), ret_type='tuple',
                                                                             recall_methods=recall_methods)
        pickle.dump(val_user_recall_item_dict,
                    open(os.path.join(save_training_path, 'val_user_recall_item_dict.pkl'), 'wb'))


def organize_train_data_multi_processing(c, is_silding_compute_sim=False, load_from_file=True, total_step=10):
    print('total_step={}'.format(total_step))
    # 1. 获取recall的结果
    compute_mode = 'once' if not is_silding_compute_sim else 'multi'
    save_training_path = os.path.join(user_data_dir, 'training', mode, compute_mode, str(c))

    save_result_train_val_path = os.path.join(save_training_path, 'train_val_label_target_id_data.pkl')
    if load_from_file and os.path.exists(save_result_train_val_path):
        return pickle.load(open(save_result_train_val_path, 'rb'))

    all_click, test_q_time = get_phase_click(c)

    click_history_df = all_click

    full_sim_pair_dict = pickle.load(open(os.path.join(save_training_path, 'full_sim_pair_dict.pkl'), 'rb'))
    step_user_recall_item_dict = pickle.load(
        open(os.path.join(save_training_path, 'step_user_recall_item_dict.pkl'), 'rb'))

    if is_silding_compute_sim:
        step_strategy_sim_pair_dict = pickle.load(
            open(os.path.join(save_training_path, 'step_strategy_sim_pair_dict.pkl'), 'rb'))
    print('read recall data done...')

    from multiprocessing import Process, JoinableQueue, Queue

    def convert(click_history_df, click_last_df, user_recall_item_dict, strategy_sim_pair_dict, input_q, result_q):
        step = input_q.get()
        print('step={} begin...'.format(step))
        user_item_time_dict = get_user_item_time_dict(click_history_df)
        # organize recall interact feat
        click_last_recall_recom_df = organize_recall_feat(user_recall_item_dict, user_item_time_dict,
                                                          strategy_sim_pair_dict, c)

        assert len(user_item_time_dict) == len(click_last_recall_recom_df['user_id'].unique()) == len(
            click_last_df['user_id'].unique())

        train_full_df = organize_label_interact_feat_df(click_last_df, click_last_recall_recom_df, c)
        train_full_df['step'] = step
        print(train_full_df['label'].value_counts())
        result_q.put(train_full_df)
        input_q.task_done()
        assert 'sim' in train_full_df.columns

    input_q = JoinableQueue()
    result_q = Queue()

    processes = []
    for step in range(total_step):
        input_q.put(step)
        click_history_df, click_last_df = get_history_and_last_click_df(click_history_df)  # override click_history_df
        user_recall_item_dict = step_user_recall_item_dict[step]
        strategy_sim_pair_dict = step_strategy_sim_pair_dict[step] if is_silding_compute_sim else full_sim_pair_dict

        processes.append(Process(target=convert, args=(click_history_df, click_last_df,
                                                       user_recall_item_dict, strategy_sim_pair_dict,
                                                       input_q, result_q)))
        processes[-1].daemon = True
        processes[-1].start()

    input_q.join()

    train_full_df_list = []
    while len(train_full_df_list) != total_step:
        train_full_df = result_q.get()
        train_full_df_list.append(train_full_df)

    for p in processes:
        p.terminate()
        p.join()

    print('obtain train data done....')

    assert len(train_full_df_list) == total_step

    if mode == 'offline':
        train_full_df = pd.concat(train_full_df_list, ignore_index=True)
        # valid data
        print('begin obtain validate data...')
        val_user_item_dict = get_user_item_time_dict(click_test)  # click_test as history
        val_user_recall_item_dict = pickle.load(
            open(os.path.join(save_training_path, 'val_user_recall_item_dict.pkl'), 'rb'))

        phase_val_last_click_answer_df = pd.read_csv('{}/{}-{}.csv'.format(offline_answer_path, infer_answer_file_prefix, c),
                                                     header=None, names=['user_id', 'item_id', 'time'])
        # organize recall interact feat
        phase_val_last_click_recall_recom_df = organize_recall_feat(val_user_recall_item_dict, val_user_item_dict,
                                                                    full_sim_pair_dict, c)

        val_full_df = organize_label_interact_feat_df(phase_val_last_click_answer_df,
                                                      phase_val_last_click_recall_recom_df, c, False)
        val_target_uids = phase_val_last_click_answer_df['user_id'].unique()

        save_train_val_path = os.path.join(save_training_path, 'train_val_label_target_id_data.pkl')
        pickle.dump([train_full_df, val_full_df, val_target_uids], open(save_train_val_path, 'wb'))

        return train_full_df, val_full_df, val_target_uids

    else:
        print('online')
        train_full_df = pd.concat(train_full_df_list, ignore_index=True)
        save_train_val_path = os.path.join(save_training_path, 'train_val_label_target_id_data.pkl')
        pickle.dump(train_full_df, open(save_train_val_path, 'wb'))
        return train_full_df


def organize_final_train_data_feat(target_phase, is_train_load_from_file=True, save_df_prefix=''):
    print('begin obtain final training data...')
    if mode == 'online':
        online_train_full_df_dict = get_glv('online_train_full_df_dict')
    else:
        train_full_df_dict = get_glv('train_full_df_dict')
        val_full_df_dict = get_glv('val_full_df_dict')

    processed_item_feat_df = get_glv('processed_item_feat_df')

    ranking_final_data = os.path.join(user_data_dir, 'ranking')
    if not os.path.exists(ranking_final_data): os.makedirs(ranking_final_data)

    train_df_path = os.path.join(ranking_final_data, save_df_prefix + 'train_final_df_phase_{}.pkl'.format(target_phase))
    val_df_path = os.path.join(ranking_final_data, save_df_prefix + 'val_final_df_phase_{}.pkl'.format(target_phase))
    w2v_path = os.path.join(ranking_final_data, save_df_prefix + 'w2v_phase_{}.pkl'.format(target_phase))

    if is_train_load_from_file and os.path.exists(train_df_path):
        print('load train from file...')
        train_final_df = pickle.load(open(train_df_path, 'rb'))
        word2vec_item_embed_dict = pickle.load(open(w2v_path, 'rb'))
        set_glv('word2vec_item_embed_dict', word2vec_item_embed_dict)
        if mode == 'offline':
            val_final_df = pickle.load(open(val_df_path, 'rb'))
            return train_final_df, val_final_df
        return train_final_df
    else:
        if mode == 'online':
            train_full_df = online_train_full_df_dict[target_phase]
            if isinstance(train_full_df, list):
                train_full_df = train_full_df[0]
        else:
            train_full_df = train_full_df_dict[target_phase]
            val_full_df = val_full_df_dict[target_phase]

        word2vec_item_embed_dict = get_word2vec_feat(train_full_df)
        set_glv('word2vec_item_embed_dict', word2vec_item_embed_dict)
        train_final_df = organize_user_item_feat(train_full_df, processed_item_feat_df,
                                                 sparse_feat, dense_feat,
                                                 is_w2v=True, word2vec_item_embed_dict=word2vec_item_embed_dict)
        pickle.dump(train_final_df[use_feats + ['label']], open(train_df_path, 'wb'))
        pickle.dump(word2vec_item_embed_dict, open(w2v_path, 'wb'))

        if mode == 'offline':
            val_final_df = organize_user_item_feat(val_full_df, processed_item_feat_df,
                                                   sparse_feat, dense_feat,
                                                   is_w2v=True, word2vec_item_embed_dict=word2vec_item_embed_dict)

            pickle.dump(val_final_df[use_feats + ['label']], open(val_df_path, 'wb'))
            return train_final_df, val_final_df

        return train_final_df


def infer_process(phase, load_from_file=True, is_use_whole_click=False,
                  is_w2v=True, is_interest=True, word2vec_item_embed_dict=None, prefix=''):
    all_click, target_infer_user_df = get_phase_click(phase)

    recall_methods = {'item-cf', 'bi-graph', 'user-cf', 'swing'}
    if is_use_whole_click:
        print('use whole click')
        phase_whole_click = get_whole_phase_click(all_click, target_infer_user_df)
        infer_user_item_time_dict = get_user_item_time_dict(phase_whole_click)
        phase_click = phase_whole_click
    else:
        infer_user_item_time_dict = get_user_item_time_dict(all_click)
        phase_click = all_click

    save_training_path = os.path.join(user_data_dir, 'recall', mode)
    sim_path = os.path.join(save_training_path, prefix + '_phase_{}_sim.pkl'.format(phase))
    recall_path = os.path.join(save_training_path, prefix + '_phase_{}.pkl'.format(phase))

    if load_from_file:
        print('load recall info from file begin, recall_path={}'.format(recall_path))
        full_sim_pair_dict = pickle.load(open(sim_path, 'rb'))
        infer_user_recall_item_dict = pickle.load(open(recall_path, 'rb'))
        print('load recall info from file done')
    else:
        item_cnt_dict = all_click.groupby('item_id')['user_id'].count().to_dict()
        user_cnt_dict = all_click.groupby('user_id')['item_id'].count().to_dict()

        full_sim_pair_dict = get_multi_source_sim_dict_results_multi_processing(phase_click,
                                                                                recall_methods=recall_methods)
        infer_user_recall_item_dict = do_multi_recall_results_multi_processing(full_sim_pair_dict,
                                                                               infer_user_item_time_dict,
                                                                               target_user_ids=target_infer_user_df['user_id'].unique(),
                                                                               ret_type='tuple', phase=phase,
                                                                               item_cnt_dict=item_cnt_dict,
                                                                               user_cnt_dict=user_cnt_dict,
                                                                               adjust_type='v2',
                                                                               recall_methods=recall_methods | {'sr-gnn'})
        pickle.dump(full_sim_pair_dict, open(sim_path, 'wb'))
        pickle.dump(infer_user_recall_item_dict, open(recall_path, 'wb'))

    infer_recall_recom_df = organize_recall_feat(infer_user_recall_item_dict, infer_user_item_time_dict,
                                                 full_sim_pair_dict, phase)

    target_infer_user_df['day_id'], target_infer_user_df['hour_id'], target_infer_user_df['minute_id'] = zip(*target_infer_user_df['time'].apply(time_info))
    infer_recall_recom_df = pd.merge(infer_recall_recom_df,
                                     target_infer_user_df[['user_id', 'time', 'day_id', 'hour_id', 'minute_id']],
                                     on='user_id', how='left')

    infer_final_df = organize_user_item_feat(infer_recall_recom_df, processed_item_feat_df,
                                             sparse_feat, dense_feat, is_interest=is_interest,
                                             is_w2v=is_w2v,word2vec_item_embed_dict=word2vec_item_embed_dict)

    return infer_recall_recom_df, infer_final_df


def organize_infer_data(target_phase, save_df_prefix, recall_prefix, is_infer_load_from_file=True):
    word2vec_item_embed_dict = get_glv('word2vec_item_embed_dict')
    ranking_final_data = os.path.join(user_data_dir, 'ranking')
    infer_df_path = os.path.join(ranking_final_data, save_df_prefix + recall_prefix + 'infer_final_df_phase_{}.pkl'.format(target_phase))

    if is_infer_load_from_file and os.path.exists(infer_df_path):
        print('load infer from file...')
        infer_recall_recom_df, infer_df = pickle.load(open(infer_df_path, 'rb'))
    else:
        infer_recall_recom_df, infer_df = infer_process(target_phase, load_from_file=True,
                                                        is_use_whole_click=True,
                                                        prefix=recall_prefix, is_interest=True,
                                                        is_w2v=True, word2vec_item_embed_dict=word2vec_item_embed_dict)
        pickle.dump([infer_recall_recom_df, infer_df[use_feats]], open(infer_df_path, 'wb'))
    return infer_recall_recom_df, infer_df
