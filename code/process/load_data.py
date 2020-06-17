from ..conf import *
import pandas as pd


def obtain_top_k_click():
    total_click = get_whole_click()
    total_click = total_click.drop_duplicates(['user_id', 'item_id', 'time'])  # important
    top50_click_np = total_click['item_id'].value_counts().index[:50].values
    top50_click = ','.join([str(i) for i in top50_click_np])
    return top50_click_np, top50_click


def get_phase_click(c):
    '''
    get click data of target phase
    :param c: target phase
    :return: all_click (includes train and test), click_q_time (infer data, i.e., user_id q_time)
    '''
    print('train_path={}, test_path={}, target_phase={}'.format(train_path, test_path, c))

    click_train = pd.read_csv('{}/{}-{}.csv'.format(train_path, train_file_prefix, c), header=None,
                              names=['user_id', 'item_id', 'time'])

    phase_test_path = "{}/{}-{}".format(test_path, test_file_prefix, c)
    click_test = pd.read_csv('{}/{}-{}.csv'.format(phase_test_path, test_file_prefix, c), header=None,
                             names=['user_id', 'item_id', 'time'])

    phase_test_path = "{}/{}-{}".format(test_path, test_file_prefix, c)
    click_q_time = pd.read_csv('{}/{}-{}.csv'.format(phase_test_path, infer_file_prefix, c), header=None,
                               names=['user_id', 'time'])

    all_click = click_train.append(click_test)

    return all_click, click_q_time


def get_online_whole_click():
    '''
    get whole click
    :return: whole click data
    '''
    whole_click = pd.DataFrame()
    for c in range(now_phase + 1):
        click_train = pd.read_csv('{}/{}-{}.csv'.format(online_train_path, train_file_prefix, c), header=None,
                                  names=['user_id', 'item_id', 'time'])
        phase_test_path = "{}/{}-{}".format(online_test_path, test_file_prefix, c)
        click_test = pd.read_csv('{}/{}-{}.csv'.format(phase_test_path, test_file_prefix, c), header=None,
                                  names=['user_id', 'item_id', 'time'])
        all_click = click_train.append(click_test)
        all_click['phase'] = c
        whole_click = whole_click.append(all_click)

    whole_click = whole_click.drop_duplicates(['user_id', 'item_id', 'time'])
    return whole_click


def get_whole_click():
    '''
    get whole click
    :return: whole click data
    '''
    print('train_path={}, test_path={}'.format(train_path, test_path))
    whole_click = pd.DataFrame()
    for c in range(now_phase + 1):
        click_train = pd.read_csv('{}/{}-{}.csv'.format(train_path, train_file_prefix, c), header=None,
                                  names=['user_id', 'item_id', 'time'])
        phase_test_path = "{}/{}-{}".format(test_path, test_file_prefix, c)
        click_test = pd.read_csv('{}/{}-{}.csv'.format(phase_test_path, test_file_prefix, c), header=None,
                                  names=['user_id', 'item_id', 'time'])
        all_click = click_train.append(click_test)
        all_click['phase'] = c
        whole_click = whole_click.append(all_click)

    print(whole_click.shape)
    whole_click = whole_click.drop_duplicates(['user_id', 'item_id', 'time'])
    print(whole_click.shape)
    return whole_click


def get_whole_phase_click(all_click, click_q_time):
    '''
    get train data for target phase (i.e., all_click) from whole click
    :param all_click: the click data of target phase
    :param click_q_time: the infer q_time of target phase
    :return: the filtered whole click data for target phase
    '''
    whole_click = get_whole_click()

    phase_item_ids = set(all_click['item_id'].unique())
    pred_user_time_dict = dict(zip(click_q_time['user_id'], click_q_time['time']))

    def group_apply_func(group_df):
        u = group_df['user_id'].iloc[0]
        if u in pred_user_time_dict:
            u_time = pred_user_time_dict[u]
            group_df = group_df[group_df['time'] <= u_time]
        return group_df

    phase_whole_click = whole_click.groupby('user_id', group_keys=False).apply(group_apply_func)
    print(phase_whole_click.head())
    print('group done')
    # filter-out the items that not in this phase
    phase_whole_click = phase_whole_click[phase_whole_click['item_id'].isin(phase_item_ids)]
    return phase_whole_click
