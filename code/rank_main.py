from code.rank import *
from code.process.feat_process import *
from code.process.convert_data import *
import code.global_variables as glv


def ranking_pipeline(target_phase, output_ranking_filename=None, model_names=['ranker'],
                     is_train_load_from_file=True, is_infer_load_from_file=True, recall_prefix='', save_df_prefix=''):
    global total_recom_lgb_df

    if mode == 'offline':
        train_final_df, val_final_df = organize_final_train_data_feat(target_phase, is_train_load_from_file, save_df_prefix)
    else:
        train_final_df = organize_final_train_data_feat(target_phase, is_train_load_from_file, save_df_prefix)
        val_final_df = None
    print('prepare train data done...')

    # load infer
    infer_recall_recom_df, infer_df = organize_infer_data(target_phase, save_df_prefix, recall_prefix,
                                                          is_infer_load_from_file=is_infer_load_from_file)
    print('prepare infer data done...')

    def gen_rec_results(output_model_name):
        global total_recom_lgb_df
        if mode == 'online':
            # check
            assert len(set(infer_recall_recom_df['user_id'].unique()) - set(
                total_recom_lgb_df[total_recom_lgb_df['phase'] == target_phase].user_id.unique())) == 0  # output
            total_recom_lgb_df = total_recom_lgb_df[total_recom_lgb_df['phase'] != target_phase]
            online_infer_recall_recom_df = infer_recall_recom_df[['user_id', 'item_id', 'prob']].rename(
                columns={'prob': 'sim'})
            online_infer_recall_recom_df['phase'] = target_phase
            total_recom_lgb_df = total_recom_lgb_df.append(online_infer_recall_recom_df)

            _, top50_click = obtain_top_k_click()
            result = get_predict(total_recom_lgb_df, 'sim', top50_click)

            result.to_csv('{}/{}-{}'.format(output_path, output_model_name, output_ranking_filename), index=False,
                          header=None)
            pickle.dump(total_recom_lgb_df,
                        open("{}/{}-{}-pkl".format(output_path, output_model_name, output_ranking_filename), 'wb'))
        print('generate rec result done...')

    if 'ranker' in model_names:
        lgb_ranker = lgb_main(train_final_df, val_final_df=val_final_df)
        lgb_rank_infer_ans = lgb_ranker.predict(infer_df[lgb_cols], axis=1)
        infer_recall_recom_df['prob'] = lgb_rank_infer_ans
        gen_rec_results('ranker')

    if 'din' in model_names:
        din_model, feature_names = din_main(target_phase, train_final_df, val_final_df)
        infer_input = {name: np.array(infer_df[name].values.tolist()) for name in feature_names}
        din_infer_ans = din_model.predict(infer_input, batch_size=BATCH_SIZE)
        infer_recall_recom_df['prob'] = din_infer_ans
        gen_rec_results('din')


if __name__ == '__main__':
    glv.init()
    item_feat_df = read_item_feat_df()
    item_content_sim_dict = get_content_sim_item(item_feat_df, topk=200)
    print(len(item_content_sim_dict))
    glv.set_glv("item_content_sim_dict", item_content_sim_dict)

    # 1. construct ranking data, this will take a bit long time !!!
    for i in range(start_phase, now_phase + 1):
        sliding_obtain_training_df(i, is_silding_compute_sim=True)

    # 2. process feat, fill in unseen items
    processed_item_feat_df, item_content_vec_dict = obtain_entire_item_feat_df()
    glv.set_glv("processed_item_feat_df", processed_item_feat_df)
    glv.set_glv("item_content_vec_dict", item_content_vec_dict)

    # encoder sparse id feat
    online_total_click = get_online_whole_click()
    sparse_feat_fit(online_total_click)

    # 3. construct training data without feat, this will take a bit long time
    if mode == 'online':
        online_train_full_df_dict = {}
        for i in range(start_phase, now_phase + 1):
            print('phase={} start'.format(i))
            if i in online_train_full_df_dict: continue
            online_train_full_df = organize_train_data_multi_processing(i, is_silding_compute_sim=True,
                                                                        load_from_file=True)
            online_train_full_df_dict[i] = online_train_full_df
        glv.set_glv("online_train_full_df_dict", online_train_full_df_dict)
    else:
        train_full_df_dict = {}
        val_full_df_dict = {}
        for i in range(start_phase, now_phase + 1):
            train_full_df, val_full_df, val_target_uids = organize_train_data_multi_processing(i,
                                                                                               is_silding_compute_sim=True,
                                                                                               load_from_file=True)
            train_full_df_dict[i] = train_full_df
            val_full_df_dict[i] = val_full_df
        glv.set_glv("train_full_df_dict", train_full_df_dict)
        glv.set_glv("val_full_df_dict", val_full_df_dict)

    global total_recom_lgb_df
    total_recom_lgb_df = sub2_df(os.path.join(output_path, 'result.csv'))

    today = time.strftime("%Y%m%d")
    output_ranking_filename = "B-ranking-{}".format(today)
    for i in range(start_phase, now_phase+1):
        print('phase={}'.format(i))
        output_ranking_filename = output_ranking_filename + "_" + str(i)
        ranking_pipeline(i, output_ranking_filename + '.csv', model_names=['ranker', 'din'],
                         is_train_load_from_file=True,
                         is_infer_load_from_file=True,
                         recall_prefix='B-recall-{}_'.format(today),
                         save_df_prefix='B-{}_'.format(today))
    # ensemble lgb+din
    ensemble(output_ranking_filename)
