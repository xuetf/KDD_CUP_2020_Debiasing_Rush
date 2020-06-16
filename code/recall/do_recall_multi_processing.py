from multiprocessing import Process, JoinableQueue, Queue
from .do_recall import *


def get_multi_source_sim_dict_results_multi_processing(history_df,
                                                       recall_methods={'item-cf', 'bi-graph', 'user-cf', 'swing'}):
    def convert(history_df, input_q, result_q):
        while True:
            name = input_q.get()
            if 'item-cf' == name:
                print('item-cf item-sim begin')
                item_sim_dict, _ = get_time_dir_aware_sim_item(history_df)
                result_q.put((name, item_sim_dict))
                print('item-cf item-sim-pair done, pair_num={}'.format(len(item_sim_dict)))

            elif 'bi-graph' == name:
                print('bi-graph item-sim begin')
                item_sim_dict, _ = get_bi_sim_item(history_df)
                result_q.put((name, item_sim_dict))
                print('bi-graph item-sim-pair done, pair_num={}'.format(len(item_sim_dict)))

            elif 'swing' == name:
                print('swing item-sim begin')
                item_sim_dict, _ = swing(history_df)
                result_q.put((name, item_sim_dict))
                print('swing item-sim-pair done, pair_num={}'.format(len(item_sim_dict)))

            elif 'user-cf' == name:
                print('user-cf user-sim begin')
                user_sim_dict, _ = get_sim_user(history_df)
                result_q.put((name, user_sim_dict))
                print('user-cf user-sim-pair done, pair_num={}'.format(len(user_sim_dict)))
            input_q.task_done()

    input_q = JoinableQueue()
    result_q = Queue()

    processes = []
    for name in recall_methods:
        input_q.put(name)
        processes.append(Process(target=convert, args=(history_df, input_q, result_q)))
        processes[-1].daemon = True
        processes[-1].start()

    input_q.join()

    recall_sim_pair_dict = {}
    while len(recall_sim_pair_dict) != len(recall_methods):
        print('current_len={}'.format(len(recall_sim_pair_dict)))
        if len(recall_sim_pair_dict) == len(recall_methods):
            break
        name, sim_pair_dict = result_q.get()
        recall_sim_pair_dict[name] = sim_pair_dict
    for p in processes:
        p.terminate()
        p.join()

    assert len(recall_sim_pair_dict) == len(recall_methods)
    return recall_sim_pair_dict


def do_multi_recall_results_multi_processing(recall_sim_pair_dict, user_item_time_dict, target_user_ids=None,
                                             ret_type='df', item_cnt_dict=None, user_cnt_dict=None, phase=None, adjust_type='v2',
                                             recall_methods={'item-cf', 'bi-graph', 'user-cf', 'swing', 'sr-gnn'}):
    from multiprocessing import Process, JoinableQueue, Queue

    print('recall-source-num={}'.format(len(recall_sim_pair_dict)))

    def convert(user_item_time_dict, target_user_ids, item_based, input_q, result_q):
        while True:
            name, sim_dict = input_q.get()
            print('do recall for {}'.format(name))
            recall_item_dict = get_recall_results(sim_dict, user_item_time_dict, target_user_ids, item_based=item_based,
                                                  item_cnt_dict=item_cnt_dict, user_cnt_dict=user_cnt_dict,
                                                  adjust_type=adjust_type)
            result_q.put((name, recall_item_dict))
            print('{} recall done, recall_user_num={}.'.format(name, len(recall_item_dict)))
            input_q.task_done()

    input_q = JoinableQueue()
    result_q = Queue()

    if target_user_ids is None:
        target_user_ids = user_item_time_dict.keys()

    processes = []
    for name, sim_dict in recall_sim_pair_dict.items():
        item_based = True if name in {'item-cf', 'bi-graph', 'swing'} else False
        input_q.put((name, sim_dict))
        processes.append(
            Process(target=convert, args=(user_item_time_dict, target_user_ids, item_based, input_q, result_q)))
        processes[-1].daemon = True
        processes[-1].start()

    input_q.join()

    recall_item_dict_list_dict = {}
    while len(recall_item_dict_list_dict) != len(recall_sim_pair_dict):
        print('current_len={}'.format(len(recall_item_dict_list_dict)))
        if len(recall_item_dict_list_dict) == len(recall_sim_pair_dict):
            break
        name, recall_item_dict = result_q.get()
        recall_item_dict_list_dict[name] = recall_item_dict

    for p in processes:
        p.terminate()
        p.join()

    print(len(recall_item_dict_list_dict))

    assert len(recall_item_dict_list_dict) == len(recall_sim_pair_dict)

    if 'sr-gnn' in recall_methods:
        print('read sr-gnn results....')
        standard_sr_gnn_recall_item_dict = read_sr_gnn_results(phase, prefix='standard', adjust_type=adjust_type)
        recall_item_dict_list_dict['sr_gnn_feat_init_v1'] = standard_sr_gnn_recall_item_dict
        print('read standard sr-gnn results done....')
        pos_weight_sr_gnn_recall_item_dict = read_sr_gnn_results(phase, prefix='pos_node_weight',
                                                                 adjust_type=adjust_type)
        recall_item_dict_list_dict['sr_gnn_pos_weight_v2'] = pos_weight_sr_gnn_recall_item_dict
        print('read pos_weight sr-gnn results done....')

    return agg_recall_results(recall_item_dict_list_dict, is_norm=True, ret_type=ret_type)
