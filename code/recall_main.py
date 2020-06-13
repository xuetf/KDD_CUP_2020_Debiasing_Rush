from code.recall.do_recall_multi_processing import *
from code.process.feat_process import *
from code.process.convert_data import *
import time

if __name__ == '__main__':
    # obtain content similarity-pairs
    global item_feat_df, item_content_sim_dict
    item_feat_df = read_item_feat_df()
    item_content_sim_dict = get_content_sim_item(item_feat_df, topk=200)
    print(len(item_content_sim_dict))

    top50_click_np, top50_click = obtain_online_top_k_click()

    recom_item = []
    total_recom_df = pd.DataFrame()
    phase_full_sim_dict = {}

    for c in range(start_phase, now_phase + 1):
        print('phase:', c)
        all_click, click_q_time = get_phase_click(c)

        phase_whole_click = get_whole_phase_click(all_click, click_q_time)
        item_cnt_dict = all_click.groupby('item_id')['user_id'].count().to_dict()
        user_cnt_dict = all_click.groupby('user_id')['item_id'].count().to_dict()

        recall_sim_pair_dict = get_multi_source_sim_dict_results_multi_processing(phase_whole_click,
                                                                                  recall_methods={'item-cf', 'bi-graph',
                                                                                                  'swing', 'user-cf'})
        user_item_time_dict = get_user_item_time_dict(phase_whole_click, is_drop_duplicated=True)

        recom_df = do_multi_recall_results_multi_processing(recall_sim_pair_dict, user_item_time_dict,
                                                            target_user_ids=click_q_time['user_id'].unique(), ret_type='df',
                                                            item_cnt_dict=item_cnt_dict, user_cnt_dict=user_cnt_dict,
                                                            phase=c, adjust_type='zjy_v1',
                                                            recall_methods={'item-cf', 'bi-graph', 'swing', 'user-cf',
                                                                            'sr-gnn'})
        recom_df['phase'] = c
        total_recom_df = total_recom_df.append(recom_df)

        phase_full_sim_dict[c] = recall_sim_pair_dict

    today = time.strftime("%Y%m%d")
    save_recall_df_as_user_tuples_dict(total_recom_df, phase_full_sim_dict,
                                       prefix='B-recall-{}'.format(today))

    # find most popular items
    result = get_predict(total_recom_df, 'sim', top50_click)
    result.to_csv(output_path + '/recall_result.csv', index=False, header=None)
