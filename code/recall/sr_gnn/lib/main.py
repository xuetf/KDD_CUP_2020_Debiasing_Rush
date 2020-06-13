from dataloader import DataLoader
from model import Model
import numpy as np
import logging
import argparse
import json
import os


def calc_recall(scores, targets, topk=10):
    is_hit = []
    for s, t in zip(scores, targets):
        is_hit.append(np.isin(t - 1, s[-topk:]))
    return np.mean(is_hit), is_hit


def calc_mrr(scores, targets, topk=10):
    mrr = []
    for s, t in zip(scores, targets):
        pos = np.where(s[-topk:] == t - 1)[0]
        if len(pos) > 0:
            mrr.append(1/(topk - pos[0]))
        else:
            mrr.append(0)
    return np.mean(mrr), mrr


def run_train(train_path, node_count, checkpoint_path, **kwargs):
    max_test_batch = kwargs.get('max_test_batch', None)
    max_len = kwargs.get('max_len', None)
    has_uid = kwargs.get("has_uid", False)
    sq_max_len = kwargs.get('sq_max_len', None)
    train = DataLoader(train_path, max_len=max_len, has_uid=has_uid, sq_max_len=sq_max_len)
    test_path = kwargs.get('test_input', None)
    test = DataLoader(test_path, True if max_test_batch else False, max_len=max_len,
                      has_uid=has_uid, sq_max_len=sq_max_len) if test_path else None

    epochs = kwargs.get('epochs', 2)
    batch_logging_step = kwargs.get('batch_logging_step', 200)
    save_step = kwargs.get('save_step', None)
    batch_size = kwargs.get('batch_size', 512)
    early_stop_epochs = kwargs.get('early_stop_epochs', None)
    if 'lr_dc' in kwargs:
        kwargs['lr_dc'] = kwargs['lr_dc'] * np.ceil(train.count / batch_size)

    if 'node_weight' in kwargs and kwargs['node_weight'].lower() in ('yes', 'true', 't', 'y', '1'):
        nw = train.get_node_weight(node_count)
        kwargs['node_weight'] = nw
        np.save(os.path.join(os.path.dirname(train_path), 'node_weight'), nw)

    logger.info('Train: {}'.format(kwargs))
    model = Model(node_count+1, checkpoint_path, **kwargs)

    global_step = int(model.run_step())
    best_result, best_epoch = [[0]*len(_at), [0]*len(_at)], [[0]*len(_at), [0]*len(_at)]  # only @20
    early_stop_counter = 0
    print_nums = lambda x: ' '.join(map(lambda num: '{:.4f}'.format(num), x))
    for epoch in range(epochs):
        slices = train.generate_batch(batch_size)
        ll = []
        logger.info('Total Batch: {}'.format(len(slices)))
        for index, i in enumerate(slices):
            adj_in, adj_out, graph_item, last_node_id, attr_dict = train.get_slice(i)
            input_session = (adj_in, adj_out, graph_item, last_node_id, attr_dict['next_item'])
            loss = model.run_train(input_session, attr_dict['node_pos'] if sq_max_len is not None else None)
            ll.append(loss)
            global_step += 1
            if index % batch_logging_step == 0:
                logger.info('Batch {}, Loss: {:.5f}'.format(index, np.mean(ll)))
            if save_step and checkpoint_path and global_step % save_step == 0:
                model.save(checkpoint_path, global_step)
        if checkpoint_path and not test:
            model.save(checkpoint_path, global_step)
        if not test:
            logger.info('Epoch: {} Train Loss: {:.4f}'.format(epoch, np.mean(ll)))
            continue

        slices = test.generate_batch(batch_size)
        test_loss_ = []
        hit = [[] for _ in _at]
        mrrs = [[] for _ in _at]
        for ii, i in enumerate(slices):
            adj_in, adj_out, graph_items, last_node_id, attr_dict = test.get_slice(i)
            input_session = (adj_in, adj_out, graph_items, last_node_id, attr_dict['next_item'])
            loss, scores = model.run_eval(input_session, attr_dict['node_pos'] if sq_max_len is not None else None)
            score_top100 = np.argsort(scores, 1)[:, -100:]
            test_loss_.append(loss)
            recall_a = [calc_recall(score_top100, attr_dict['next_item'], k) for k in _at]
            mrr_a = [calc_mrr(score_top100, attr_dict['next_item'], k) for k in _at]
            recall = map(lambda x: x[0], recall_a)
            mrr = map(lambda x: x[0], mrr_a)
            logger.info('Test Loss: {:.4f}  @{}, Recall: {}  MRR: {}'.format(loss, ' '.join(map(str, _at)), print_nums(recall), print_nums(mrr)))
            for pos, _ in enumerate(_at):
                hit[pos] += recall_a[pos][1]
                mrrs[pos] += mrr_a[pos][1]
            if max_test_batch and ii >= max_test_batch - 1: break
        epoch_hits = np.mean(hit, axis=1)
        epoch_mrr = np.mean(mrrs, axis=1)
        is_improve = 0
        for i in range(len(_at)):
            if epoch_hits[i] > best_result[0][i]:
                best_result[0][i] = epoch_hits[i]
                best_epoch[0][i] = epoch
                is_improve = 1
            if epoch_mrr[i] > best_result[1][i]:
                best_result[1][i] = epoch_mrr[i]
                best_epoch[1][i] = epoch
                is_improve = 1
        logger.info('Epoch: {} Train Loss: {:.4f} Test Loss: {:.4f} Recall: {} MRR: {}'.format(epoch, np.mean(ll), np.mean(test_loss_), print_nums(epoch_hits), print_nums(epoch_mrr)))
        logger.info('Best Recall and MRR: {},  {}  Epoch: {},  {}'.format(print_nums(best_result[0]),
                                                                          print_nums(best_result[1]),
                                                                          ' '.join(map(str, best_epoch[0])),
                                                                          ' '.join(map(str, best_epoch[1]))))
        if checkpoint_path and is_improve == 1:
            model.save(checkpoint_path, global_step)
        early_stop_counter += 1 - is_improve
        if early_stop_epochs and early_stop_counter >= early_stop_epochs:
            logger.info('After {} epochs not improve, early stop'.format(early_stop_counter))
            break
    logger.info('Best Recall and MRR: {},  {}  Epoch: {},  {}'.format(print_nums(best_result[0]),
                                                                      print_nums(best_result[1]),
                                                                      ' '.join(map(str, best_epoch[0])),
                                                                      ' '.join(map(str, best_epoch[1]))))


def run_eval(eval_path, node_count, checkpoint_path, **kwargs):
    max_test_batch = kwargs.get('max_test_batch', None)
    max_len = kwargs.get('max_len', None)
    has_uid = kwargs.get("has_uid", False)
    sq_max_len = kwargs.get('sq_max_len', None)
    eval_data = DataLoader(eval_path, True if max_test_batch else False, max_len=max_len,
                           has_uid=has_uid, sq_max_len=sq_max_len)

    batch_size = kwargs.get('batch_size', 512)
    model = Model(node_count+1, checkpoint_path, **kwargs)

    slices = eval_data.generate_batch(batch_size)
    logger.info('Total Batch: {}'.format(len(slices)))
    test_loss_ = []

    hit = [[] for _ in _at]
    mrrs = [[] for _ in _at]
    print_nums = lambda x: ' '.join(map(lambda num: '{:.4f}'.format(num), x))
    for ii, i in enumerate(slices):
        adj_in, adj_out, graph_item, last_node_id, attr_dict = eval_data.get_slice(i)
        input_session = (adj_in, adj_out, graph_item, last_node_id, attr_dict['next_item'])
        loss, scores = model.run_eval(input_session, attr_dict['node_pos'] if sq_max_len is not None else None)
        score_top100 = np.argsort(scores, 1)[:, -100:]
        test_loss_.append(loss)
        recall_a = [calc_recall(score_top100, attr_dict['next_item'], k) for k in _at]
        mrr_a = [calc_mrr(score_top100, attr_dict['next_item'], k) for k in _at]
        recall = map(lambda x: x[0], recall_a)
        mrr = map(lambda x: x[0], mrr_a)
        logger.info(
            'Eval Loss: {:.4f}  @{}, Recall: {}  MRR: {}'.format(loss, ' '.join(map(str, _at)), print_nums(recall),
                                                                 print_nums(mrr)))
        for pos, _ in enumerate(_at):
            hit[pos] += recall_a[pos][1]
            mrrs[pos] += mrr_a[pos][1]
        if max_test_batch and ii >= max_test_batch - 1: break
    logger.info('Total: Eval Loss: {:.4f} Recall: {} MRR: {}'.format(np.mean(test_loss_), print_nums(np.mean(hit, axis=1)), print_nums(np.mean(mrrs, axis=1))))


def run_recommend(session_input, checkpoint_path, node_count, output_path, **kwargs):
    max_len = kwargs.get('max_len', None)
    has_uid = kwargs.get("has_uid", False)
    sq_max_len = kwargs.get('sq_max_len', None)
    session_data = DataLoader(session_input, False, False, True,
                              max_len=max_len, has_uid=has_uid, sq_max_len=sq_max_len)
    batch_size = kwargs.get('batch_size', 512)
    if 'node_weight' in kwargs and kwargs['node_weight'].lower() in ('yes', 'true', 't', 'y', '1') \
            and kwargs.get('node_weight_trainable', False):
        nw = np.zeros([node_count + 1], np.float32)
        kwargs['node_weight'] = nw
    model = Model(node_count+1, checkpoint_path, **kwargs)
    id_lookup_file = kwargs.get('item_lookup', None)
    if id_lookup_file:
        item_id = {}
        with open(id_lookup_file, 'r') as f:
            for line in f:
                num_id, vid = line.split()
                item_id[int(num_id)] = vid
    slices = session_data.generate_batch(batch_size)
    logger.info('Total Batch: {}'.format(len(slices)))
    total_users = 0
    remove_duplicates = kwargs.get('remove_duplicates', None)
    if remove_duplicates is None: remove_duplicates = True
    rec_count = kwargs.get("rec_count", 100)
    rec_extra_count = kwargs.get("rec_extra_count", 50)
    topks = rec_count + rec_extra_count + 1
    with open(output_path, 'w') as f:
        for i, batch in enumerate(slices):
            adj_in, adj_out, graph_item, last_node_id, attr_dict = session_data.get_slice(batch)
            input_sessions = (adj_in, adj_out, graph_item, last_node_id)
            scores = model.run_predict(input_sessions, attr_dict['node_pos'] if sq_max_len is not None else None)
            score_topks = np.argsort(scores, 1)[:, :-topks:-1]
            topk_logits = np.asarray([scores[k][score_topks[k]] for k in range(len(score_topks))])
            score_topks = score_topks + 1
            headers = attr_dict['header']
            for j in range(len(score_topks)):
                items = score_topks[j].tolist()
                logits = np.exp(topk_logits[j] - np.max(topk_logits[j]))
                ls = np.sum(logits)
                logits = (logits/ls).tolist()
                item_score = list(zip(items, logits))
                if remove_duplicates:
                    session_items = set(graph_item[j])
                    item_score = list(filter(lambda x: x[0] not in session_items, item_score))
                if id_lookup_file:
                    item_score = list(map(lambda x: (item_id[x[0]], x[1]), item_score))
                item_score = item_score[:rec_count]
                f.write(json.dumps([headers[j], item_score])+'\n')
                total_users += 1
            if i % 200 == 0:
                logger.info('Batch {} Finished, users: {}'.format(i, total_users))
        logger.info('Recommend Finished, users: {}'.format(total_users))


def run_node_embedding(checkpoint_path, node_count, output_path, **kwargs):
    if 'node_weight' in kwargs and kwargs['node_weight'].lower() in ('yes', 'true', 't', 'y', '1'):
        nw = np.zeros([node_count+1], np.float32)
        kwargs['node_weight'] = nw
    model = Model(node_count+1, checkpoint_path, **kwargs)
    np.save(output_path, model.run_embedding()[0])


def get_arg_parser():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', dest='task', required=True, choices=task, type=str)
    parser.add_argument('--node_count', dest='node_count', required=True, type=int)
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', required=True, type=str)

    parser.add_argument('--l2', dest='l2', required=False, type=float)
    parser.add_argument('--lr', dest='lr', required=False, type=float)
    parser.add_argument('--gru_step', dest='gru_step', required=False, type=int)
    parser.add_argument('--batch_size', dest='batch_size', required=False, type=int, default=512)
    parser.add_argument('--hidden_size', dest='hidden_size', required=False, type=int)
    parser.add_argument('--epochs', dest='epochs', required=False, type=int)
    parser.add_argument('--batch_logging_step', dest='batch_logging_step', required=False, type=int)
    parser.add_argument('--save_step', dest='save_step', required=False, type=int)
    parser.add_argument('--max_test_batch', dest='max_test_batch', required=False, type=int)
    parser.add_argument('--lr_dc', dest='lr_dc', type=int, required=False)
    parser.add_argument('--dc_rate', dest='dc_rate', type=float, required=False)
    parser.add_argument('--early_stop_epochs', dest='early_stop_epochs', required=False, type=int)
    parser.add_argument('--sigma', dest='sigma', required=False, type=float)
    parser.add_argument('--max_len', dest='max_len', required=False, type=int)
    parser.add_argument('--has_uid', dest='has_uid', required=False, type=str2bool)
    parser.add_argument('--feature_init', dest='feature_init', required=False, type=str)
    parser.add_argument('--node_weight', dest='node_weight', required=False, type=str)
    parser.add_argument('--node_weight_trainable', dest='node_weight_trainable', required=False, type=str2bool)
    parser.add_argument('--sq_max_len', dest='sq_max_len', required=False, type=int)

    parser.add_argument('--train_input', dest='train_input', required=False, type=str)
    parser.add_argument('--test_input', dest='test_input', required=False, type=str)
    parser.add_argument('--eval_input', dest='eval_input', required=False, type=str)
    parser.add_argument('--session_input', dest='session_input', required=False, type=str)
    parser.add_argument('--item_lookup', dest='item_lookup', required=False, type=str)
    parser.add_argument('--item_feature', dest='item_feature', required=False, type=str)
    parser.add_argument('--recommend_output', dest='recommend_output', required=False, type=str)
    parser.add_argument('--embedding_output', dest='embedding_output', required=False, type=str)
    parser.add_argument('--rec_extra_count', dest='rec_extra_count', required=False, type=int)
    parser.add_argument('--rec_count', dest='rec_count', required=False, type=int)
    parser.add_argument('--remove_duplicates', dest='remove_duplicates', required=False, type=str2bool)
    return parser


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s", level=logging.INFO)
    logger = logging.getLogger("main")
    _at = [50]
    task = ['train', 'eval', 'recommend', 'node_embedding']
    required_args = ['node_count', 'checkpoint_path', 'task']
    args = get_arg_parser().parse_args()
    arg_node_count = args.node_count
    arg_checkpoint = args.checkpoint_path
    arg_task = args.task
    arg_dict = {k: v for k, v in args.__dict__.items() if k not in required_args and v is not None}
    print(arg_dict)
    if arg_task == 'train':
        if not args.train_input:
            logger.error("Arg Error: --train_input")
            exit(-1)
        arg_dict.pop('train_input')
        run_train(args.train_input, arg_node_count, arg_checkpoint, **arg_dict)
    elif arg_task == 'eval':
        if not args.eval_input:
            logger.error("Arg Error: --eval_input")
            exit(-1)
        arg_dict.pop('eval_input')
        run_eval(args.eval_input, arg_node_count, arg_checkpoint, **arg_dict)
    elif args.task == 'recommend':
        if not args.recommend_output or not args.session_input:
            logger.error("Arg Error: --recommend_output/--session_input")
            exit(-1)
        arg_dict.pop('session_input')
        arg_dict.pop('recommend_output')
        run_recommend(args.session_input, arg_checkpoint, arg_node_count, args.recommend_output, **arg_dict)
    elif args.task == 'node_embedding':
        if not args.embedding_output:
            logger.error("Arg Error: --embedding_output")
            exit(-1)
        arg_dict.pop('embedding_output')
        run_node_embedding(arg_checkpoint, arg_node_count, args.embedding_output, **arg_dict)
