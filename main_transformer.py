import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from data import sequence
from modules import models
from utils import utility, metrics, earlystopping, log_tools, plot_tools
import random
from collections import Counter


device = torch.device("cuda:0")  # 使用第一个CUDA设备




def evaluation(args, model, eval_loader, threshold_moving=True, thres=0.5, device='cpu'):
    model.to(device)
    model.eval()
    prob_list = []
    label_list = []
    with torch.no_grad():
        for (batch_seq, batch_labels,batch_indices) in tqdm(eval_loader):
            batch_seq = batch_seq.to(device)
            #batch_logits = model(batch_seq)
            batch_indices = batch_indices.to(device)

            # 将批次数据和索引传递给模型
            batch_logits = model(batch_seq, batch_indices)
            prob_list.append(batch_logits.cpu())
            label_list.append(batch_labels.cpu())

    # shape=(len(eval_loader), 2)
    probs = torch.cat(prob_list, dim=0)

    # for label alighment when using train_loader
    eval_labels = torch.cat(label_list, dim=0)

    return metrics.convert_probs(eval_labels, probs, threshold_moving=threshold_moving, thres=thres)


def train(args, data, dataset, run_id):
    if args['gpu'] < 0:
        device = 'cpu'
    else:
        device = f"cuda:{args['gpu']}"

    eval_device = 'cpu' if args['cpu_eval'] else device


    seq_data, labels, train_nid, val_nid, test_nid, feat_dim, n_classes, n_relations, pseudo_labels,original_train_nid = data

    train_set, val_set, test_set = dataset


    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True,
                              drop_last=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False,
                            drop_last=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False,
                             drop_last=False, num_workers=0)

    random_seed = 717

    model = models.TransformerEncoderNet(feat_dim=feat_dim, emb_dim=args['emb_dim'],
                                         n_classes=n_classes, n_hops=args['n_hops'],
                                         n_relations=n_relations, dim_feedforward=args['ff_dim'],
                                         n_layers=args['n_layers'], n_heads=args['n_heads'],dropout=args['dropout'], seed=random_seed)

    model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    # log tools
    summary = log_tools.SummaryBox(task_name=f"{args['dataset']}", flush_secs=args['flush_seconds'],
                                   log_dir=args['log_dir'])

    summary.save_config(args)

    if args['early_stop'] > 0:
        stopper = earlystopping.EarlyStopper(patience=args['early_stop'],
                                             dataset_name=args['dataset'],
                                             start_wall_time=summary.start_wall_time,
                                             log_dir=args['log_dir'])

    timer = log_tools.Timer(task_name=f"Train on {args['dataset']}")




    for epoch in range(args['max_epochs']):
        print(f"Train on epoch {epoch:>4d}:")
        model.train()
        timer.start()

        total_loss = 0.0

        for step, (batch_seq, batch_labels,batch_indices) in enumerate(tqdm(train_loader)):
            batch_seq = batch_seq.to(device)
            batch_labels = batch_labels.to(device)
            batch_indices = batch_indices.to(device)


            batch_logits = model(batch_seq, batch_indices)


            loss = loss_func(batch_logits, batch_labels)
            total_loss += loss

            summary.update_loss(loss, global_step=epoch *
                                len(train_loader) + step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        timer.end()

        if epoch % args['eval_interval'] == 0:
            print(f"AVG. loss={total_loss / len(train_loader): 3.4f}, "
                  f"Elapsed time={timer.avg_time:.2f}(s)")

            val_true, val_prob, val_pred = evaluation(
                args, model, val_loader, device=eval_device)
            results = metrics.eval_model(val_true, val_prob, val_pred)

            summary.update_metrics(results, global_step=epoch)

            if args['early_stop'] > 0:
                if stopper.step(results.auc_gnn, epoch, model):
                    break

    summary.close()

    print("\nBest Epoch {}, Val {:.4f}".format(
        stopper.best_ep, stopper.best_score))

    if args['early_stop']:
        stopper.load_checkpoint(model)


        val_true, val_prob, val_pred = evaluation(
            args, model, val_loader, device=eval_device)
        val_results = metrics.eval_model(val_true, val_prob, val_pred)

        summary.add_figure(figure=plot_tools.plot_roc_curve(val_true, val_prob),
                           fig_name=f"ROC-AUC Curve ({args['dataset']}_{run_id})")
        summary.add_figure(figure=plot_tools.plot_pr_curve(val_true, val_prob),
                           fig_name=f"PR Curve ({args['dataset']}_{run_id})")

        print(f"best_roc_thres: {val_results.best_roc_thres} \n"
              f"best_pr_thres: {val_results.best_pr_thres}")
        te_true, te_prob, te_pred = evaluation(args, model, test_loader,
                                               thres=val_results.best_pr_thres,
                                               device=eval_device)
        results = metrics.eval_model(te_true, te_prob, te_pred)

    return [results.f1_macro, results.auc_gnn, results.gmean_gnn,
            results.precision_1, results.recall_1, results.ap_gnn,
            results.f1_binary_1, results.f1_binary_0, results.recall_macro,
            stopper.best_ep]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAGA")

    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config.')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='<dir> to store train logs.')
    parser.add_argument('--early_stop', type=int, default=30,
                        help='The patience when using early stop.\n'
                             'Default: 30, 0 disables earlystopper.')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Which gpu to use 0/1/..., -1 using only cpu')
    parser.add_argument('--n_workers', type=int, default=0,
                        help='Number of extra processes for dataloader.')

    parser.add_argument('--n_runs', type=int, default=1,
                        help='Repeat the training n times.')
    parser.add_argument('--seed', type=int, default=114514,
                        help='Random seed for reproducibility.')


    args = vars(parser.parse_args())
    train_config = utility.load_config(args['config'])
    args = utility.setup_args(args, train_config)

    set_seed(args['seed'])

    # load input sequence preprocessed by graph2seq_mp.py
    data = sequence.load_sequence_data(args)

    seq_data, labels, train_nid, val_nid, test_nid, feat_dim, \
        n_classes, n_relations ,pseudo_labels ,original_train_nid = data

    dataset = sequence.split_dataset(data)


    result_list = []
    for i in range(args['n_runs']):
        res = train(args, data, dataset, i)
        result_list.append(res)

    print(result_list)
    metrics.calc_mean_sd(result_list)
