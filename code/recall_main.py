from code.recall.do_recall_multi_processing import *
from code.process.feat_process import *
from code.process.convert_data import *
import time
import code.global_variables as glv


if __name__ == '__main__':
    if mode == 'offline':
        from code.process.tr_val_split import *
        from code.process.evaluate import *
        tr_val_split()
        print('train/validate split done...')
        create_answer_file_for_evaluation(output_answer_fname=output_path +'/debias_track_answer.csv')
        print('create offline eval answer done...')

    glv.init()  # init global variable

    # obtain content similarity-pairs
    item_feat_df = read_item_feat_df()
    item_content_sim_dict = get_content_sim_item(item_feat_df, topk=200)
    glv.set_glv("item_content_sim_dict", item_content_sim_dict)
    print(len(item_content_sim_dict))

    top50_click_np, top50_click = obtain_top_k_click()

    recom_item = []
    total_recom_df = pd.DataFrame()
    phase_full_sim_dict = {}

    cf_methods = {'item-cf', 'bi-graph', 'swing', 'user-cf'}
    # setup whether to use multi-processing
    if is_multi_processing:
        print('using multi_processing')
        do_cf_sim_func = get_multi_source_sim_dict_results_multi_processing
        do_recall_func = do_multi_recall_results_multi_processing
    else:
        print('using single_processing')
        do_cf_sim_func = get_multi_source_sim_dict_results
        do_recall_func = do_multi_recall_results

    for c in range(start_phase, now_phase + 1):
        print('phase:', c)
        all_click, click_q_time = get_phase_click(c)

        if mode == 'online':
            phase_click = get_whole_phase_click(all_click, click_q_time)
        else:
            phase_click = all_click

        item_cnt_dict = all_click.groupby('item_id')['user_id'].count().to_dict()
        user_cnt_dict = all_click.groupby('user_id')['item_id'].count().to_dict()

        recall_sim_pair_dict = do_cf_sim_func(phase_click, recall_methods=cf_methods)

        user_item_time_dict = get_user_item_time_dict(phase_click, is_drop_duplicated=True)

        recom_df = do_recall_func(recall_sim_pair_dict, user_item_time_dict,
                                  target_user_ids=click_q_time['user_id'].unique(), ret_type='df',
                                  item_cnt_dict=item_cnt_dict, user_cnt_dict=user_cnt_dict,
                                  phase=c, adjust_type='v2', recall_methods=cf_methods )

        recom_df['phase'] = c
        total_recom_df = total_recom_df.append(recom_df)

        phase_full_sim_dict[c] = recall_sim_pair_dict

    today = time.strftime("%Y%m%d")
    save_recall_df_as_user_tuples_dict(total_recom_df, phase_full_sim_dict,
                                       prefix='B-recall-{}'.format(today))

    result = get_predict(total_recom_df, 'sim', top50_click)
    result.to_csv(output_path + '/result.csv', index=False, header=None)

    if mode == 'offline':
        # eval offline
        evaluate(output_path + '/result.csv',
                 answer_fname=output_path+'/debias_track_answer.csv')


