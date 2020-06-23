# coding=utf-8
from __future__ import division
from __future__ import print_function
import datetime
from collections import defaultdict

import numpy as np
from ..conf import *


# the higher scores, the better performance
def evaluate_each_phase(predictions, answers):
    list_item_degress = []
    for user_id in answers:
        item_id, item_degree = answers[user_id]
        list_item_degress.append(item_degree)
    list_item_degress.sort()
    median_item_degree = list_item_degress[len(list_item_degress) // 2]

    num_cases_full = 0.0
    ndcg_50_full = 0.0
    ndcg_50_half = 0.0
    num_cases_half = 0.0
    hitrate_50_full = 0.0
    hitrate_50_half = 0.0
    for user_id in answers:
        item_id, item_degree = answers[user_id]
        rank = 0
        while rank < 50 and predictions[user_id][rank] != item_id:
            rank += 1
        num_cases_full += 1.0
        if rank < 50:
            ndcg_50_full += 1.0 / np.log2(rank + 2.0)
            hitrate_50_full += 1.0
        if item_degree <= median_item_degree:
            num_cases_half += 1.0
            if rank < 50:
                ndcg_50_half += 1.0 / np.log2(rank + 2.0)
                hitrate_50_half += 1.0
    ndcg_50_full /= num_cases_full
    hitrate_50_full /= num_cases_full
    ndcg_50_half /= num_cases_half
    hitrate_50_half /= num_cases_half
    return np.array([ndcg_50_full, ndcg_50_half,
                     hitrate_50_full, hitrate_50_half], dtype=np.float32)


# submit_fname is the path to the file submitted by the participants.
# debias_track_answer.csv is the standard answer, which is not released.
def evaluate(submit_fname,
             answer_fname='debias_track_answer.csv', current_time=None):
    schedule_in_unix_time = [
        0,  # ........ 1970-01-01 08:00:00 (T=0)
        1586534399,  # 2020-04-10 23:59:59 (T=1)
        1587139199,  # 2020-04-17 23:59:59 (T=2)
        1587743999,  # 2020-04-24 23:59:59 (T=3)
        1588348799,  # 2020-05-01 23:59:59 (T=4)
        1588953599,  # 2020-05-08 23:59:59 (T=5)
        1589558399,  # 2020-05-15 23:59:59 (T=6)
        1590163199,  # 2020-05-22 23:59:59 (T=7)
        1590767999,  # 2020-05-29 23:59:59 (T=8)
        1591372799  # .2020-06-05 23:59:59 (T=9)
    ]
    assert len(schedule_in_unix_time) == 10
    for i in range(1, len(schedule_in_unix_time) - 1):
        # 604800 == one week
        assert schedule_in_unix_time[i] + 604800 == schedule_in_unix_time[i + 1]

    if current_time is None:
        current_time = int(time.time())
    print('current_time:', current_time)
    print('date_time:', datetime.datetime.fromtimestamp(current_time))
    current_phase = 0
    while (current_phase < 9) and (
            current_time > schedule_in_unix_time[current_phase + 1]):
        current_phase += 1
    print('current_phase:', current_phase)

    try:
        answers = [{} for _ in range(10)]
        with open(answer_fname, 'r') as fin:
            for line in fin:
                line = [int(x) for x in line.split(',')]
                phase_id, user_id, item_id, item_degree = line
                if mode == 'online':
                    assert user_id % 11 == phase_id
                # exactly one test case for each user_id
                answers[phase_id][user_id] = (item_id, item_degree)
    except Exception as e:
        return print('server-side error: answer file incorrect', e)

    try:
        predictions = {}
        with open(submit_fname, 'r') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue
                line = line.split(',')
                user_id = int(line[0])
                if user_id in predictions:
                    return print('submitted duplicate user_ids')
                item_ids = [int(i) for i in line[1:]]
                if len(item_ids) != 50:
                    return print('each row need have 50 items')
                if len(set(item_ids)) != 50:
                    return print('each row need have 50 DISTINCT items')
                predictions[user_id] = item_ids
    except Exception as e:
        return print('submission not in correct format,e={}'.format(e))

    scores = np.zeros(4, dtype=np.float32)

    # The final winning teams will be decided based on phase T=7,8,9 only.
    # We thus fix the scores to 1.0 for phase 0,1,2,...,6 at the final stage.
    if current_phase >= 7:  # if at the final stage, i.e., T=7,8,9
        scores += 7.0  # then fix the scores to 1.0 for phase 0,1,2,...,6
    phase_beg = (7 if (current_phase >= 7) else 0)
    phase_end = current_phase + 1
    for phase_id in range(phase_beg, phase_end):
        for user_id in answers[phase_id]:
            if user_id not in predictions:
                return print('user_id %d of phase %d not in submission' % (
                    user_id, phase_id))
        try:
            # We sum the scores from all the phases, instead of averaging them.
            phase_score = evaluate_each_phase(predictions, answers[phase_id])
            print('phase_id={}, score={}'.format(phase_id, phase_score))
            scores += phase_score
        except Exception as e:
            return print('error occurred during evaluation, e={}'.format(e))

    print("score={},\nhitrate_50_full={},\nndcg_50_full={},\nhitrate_50_half={}, \nndcg_50_half={}".format(
        float(scores[0]), float(scores[2]), float(scores[0]), float(scores[3]), float(scores[1])
    ))
    return scores[0]
    # return report_score(
    #     stdout, score=float(scores[0]),
    #     ndcg_50_full=float(scores[0]), ndcg_50_half=float(scores[1]),
    #     hitrate_50_full=float(scores[2]), hitrate_50_half=float(scores[3]))


# FYI. You can create a fake answer file for validation based on this. For example,
# you can mask the latest ONE click made by each user in underexpose_test_click-T.csv,
# and use those masked clicks to create your own validation set, i.e.,
# a fake underexpose_test_qtime_with_answer-T.csv for validation.
def create_answer_file_for_evaluation(output_answer_fname='debias_track_answer.csv'):
    train = train_path + '/underexpose_train_click-%d.csv'
    test = test_path + '/underexpose_test_click-%d/underexpose_test_click-%d.csv'

    # underexpose_test_qtime-T.csv contains only <user_id, time>
    # underexpose_test_qtime_with_answer-T.csv contains <user_id, item_id, time>
    answer = offline_answer_path + '/underexpose_test_qtime_with_answer-%d.csv'  # not released

    item_deg = defaultdict(lambda: 0)
    with open(output_answer_fname, 'w') as fout:
        for phase_id in range(now_phase + 1):
            with open(train % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    item_deg[item_id] += 1
            with open(test % (phase_id, phase_id)) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    item_deg[item_id] += 1
            with open(answer % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    if mode == 'online':
                        assert user_id % 11 == phase_id
                    print(phase_id, user_id, item_id, item_deg[item_id],
                          sep=',', file=fout)


# submit_fname is the path to the file submitted by the participants.
# debias_track_answer.csv is the standard answer, which is not released.
def evaluate_target_phase(submit_fname, target_phase,
                          answer_fname='debias_track_answer.csv', current_time=None):
    schedule_in_unix_time = [
        0,  # ........ 1970-01-01 08:00:00 (T=0)
        1586534399,  # 2020-04-10 23:59:59 (T=1)
        1587139199,  # 2020-04-17 23:59:59 (T=2)
        1587743999,  # 2020-04-24 23:59:59 (T=3)
        1588348799,  # 2020-05-01 23:59:59 (T=4)
        1588953599,  # 2020-05-08 23:59:59 (T=5)
        1589558399,  # 2020-05-15 23:59:59 (T=6)
        1590163199,  # 2020-05-22 23:59:59 (T=7)
        1590767999,  # 2020-05-29 23:59:59 (T=8)
        1591372799  # .2020-06-05 23:59:59 (T=9)
    ]
    assert len(schedule_in_unix_time) == 10
    for i in range(1, len(schedule_in_unix_time) - 1):
        # 604800 == one week
        assert schedule_in_unix_time[i] + 604800 == schedule_in_unix_time[i + 1]

    if current_time is None:
        current_time = int(time.time())
    print('current_time:', current_time)
    print('date_time:', datetime.datetime.fromtimestamp(current_time))
    current_phase = 0
    while (current_phase < 9) and (
            current_time > schedule_in_unix_time[current_phase + 1]):
        current_phase += 1
    print('current_phase:', current_phase)

    try:
        answers = [{} for _ in range(10)]
        with open(answer_fname, 'r') as fin:
            for line in fin:
                line = [int(x) for x in line.split(',')]
                phase_id, user_id, item_id, item_degree = line
                # assert user_id % 11 == phase_id
                # exactly one test case for each user_id
                answers[phase_id][user_id] = (item_id, item_degree)
    except Exception as e:
        return print('server-side error: answer file incorrect', e)

    try:
        predictions = {}
        with open(submit_fname, 'r') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue
                line = line.split(',')
                user_id = int(line[0])
                if user_id in predictions:
                    return print('submitted duplicate user_ids')
                item_ids = [int(i) for i in line[1:]]
                if len(item_ids) != 50:
                    return print('each row need have 50 items')
                if len(set(item_ids)) != 50:
                    return print('each row need have 50 DISTINCT items')
                predictions[user_id] = item_ids
    except Exception as e:
        return print('submission not in correct format,e={}'.format(e))

    scores = np.zeros(4, dtype=np.float32)

    # The final winning teams will be decided based on phase T=7,8,9 only.
    # We thus fix the scores to 1.0 for phase 0,1,2,...,6 at the final stage.
    if current_phase >= 7:  # if at the final stage, i.e., T=7,8,9
        scores += 7.0  # then fix the scores to 1.0 for phase 0,1,2,...,6
    phase_beg = (7 if (current_phase >= 7) else 0)
    phase_end = current_phase + 1
    for phase_id in [target_phase]:
        for user_id in answers[phase_id]:
            if user_id not in predictions:
                return print('user_id %d of phase %d not in submission' % (
                    user_id, phase_id))
        try:
            # We sum the scores from all the phases, instead of averaging them.
            phase_score = evaluate_each_phase(predictions, answers[phase_id])
            print('phase_id={}, score={}'.format(phase_id, phase_score))
            scores += phase_score
        except Exception as e:
            return print('error occurred during evaluation e={}'.format(e))

    print("score={},\nhitrate_50_full={},\nndcg_50_full={},\nhitrate_50_half={}, \nndcg_50_half={}".format(
        float(scores[0]), float(scores[2]), float(scores[0]), float(scores[3]), float(scores[1])
    ))
    return scores[0]
    # return report_score(
    #     stdout, score=float(scores[0]),
    #     ndcg_50_full=float(scores[0]), ndcg_50_half=float(scores[1]),
    #     hitrate_50_full=float(scores[2]), hitrate_50_half=float(scores[3]))
