import argparse
import os
import numpy as np
import dgl
import torch
import time
from tqdm import tqdm

import data_utils

import logging




def save_sequence(args, data):
    pass


def graph2seq(args, graph_data):
    group_loader = data_utils.GroupFeatureSequenceLoader(graph_data)
    g = graph_data.graph
    labels = graph_data.labels
    train_nid = graph_data.train_nid
    original_train_nid = graph_data.original_train_nid
    val_nid = graph_data.val_nid
    test_nid = graph_data.test_nid
    n_classes = graph_data.n_classes
    pseudo_labels = graph_data.pseudo_labels

    output_file = "node_ids.txt"

    # 打开文件以追加写入
    with open(output_file, "w") as file:
        # 保存训练集节点标识
        file.write("Train Nodes:\n")
        for i in range(0, len(train_nid), 100):
            batch = train_nid[i:i + 100]
            for nid in batch:
                file.write(str(nid) + "\n")
            file.write("\n")  # 每100个节点后添加空行

        # 保存验证集节点标识
        file.write("Validation Nodes:\n")
        for i in range(0, len(val_nid), 100):
            batch = val_nid[i:i + 100]
            for nid in batch:
                file.write(str(nid) + "\n")
            file.write("\n")  # 每100个节点后添加空行

        # 保存测试集节点标识
        file.write("Test Nodes:\n")
        for i in range(0, len(test_nid), 100):
            batch = test_nid[i:i + 100]
            for nid in batch:
                file.write(str(nid) + "\n")
            file.write("\n")  # 每100个节点后添加空行

    feat_dim = graph_data.feat_dim
    n_relations = graph_data.n_relations
    n_groups = n_classes + 1
    n_hops = len(args['fanouts'])
    n_nodes = g.num_nodes()

    seq_len = n_relations * (n_hops * n_groups )

    all_nid = g.nodes()
    output_file = "all_node_ids.txt"

    with open(output_file, "w") as file:
        # 保存节点标识
        file.write("All Nodes:\n")
        node_list = list(all_nid)
        for i in range(0, len(node_list), 100):
            batch = node_list[i:i + 100]
            for nid in batch:
                file.write(str(nid) + "\n")
            file.write("\n")  # 每100个节点后添加空行

    file_dir = os.path.join(args['save_dir'], args['dataset'])
    os.makedirs(file_dir, exist_ok=True)

    infos = np.array([feat_dim, n_classes, n_relations], dtype=np.int64)
    
    info_name = f"{args['dataset']}_infos_" \
                f"{args['train_size']}_{args['val_size']}_{args['seed']}.npz"
    info_file = os.path.join(file_dir, info_name)
    print(f"Saving infos to {info_file}")
    np.savez(info_file, label=labels.numpy(), pseudo_labels= pseudo_labels.numpy(),
             original_train_nid = original_train_nid,train_nid=train_nid, val_nid=val_nid,
             test_nid=test_nid, infos=infos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='graph2seq')
    parser.add_argument('--dataset', type=str, default='amazon',
                        help='Dataset name, [amazon, yelp, BF10M]')

    parser.add_argument('--train_size', type=float, default=0.4,
                        help='Train size of nodes.')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Val size of nodes.')
    parser.add_argument('--seed', type=int, default=717,
                        help='Collecting neighbots in n hops.')

    parser.add_argument('--norm_feat', action='store_true', default=False,
                        help='Using group norm, default False')
    parser.add_argument('--grp_norm', action='store_true', default=False,
                        help='Using group norm, default False')
    parser.add_argument('--force_reload', action='store_true', default=False,
                        help='Using group norm, default False')
    parser.add_argument('--add_self_loop', action='store_true', default=False,
                    help='add self-loop to all the nodes')
    parser.add_argument('--i_1', type=int, default=0, help='Number of additional label 1 nodes.')
    parser.add_argument('--i_0', type=int, default=0, help='Number of additional label 0 nodes.')
    parser.add_argument('--fanouts', type=int, default=[-1], nargs='+',
                        help='Sampling neighbors, default [-1] means full neighbors')

    parser.add_argument('--base_dir', type=str, default='~/.dgl',
                        help='Directory for loading graph data.')
    parser.add_argument('--save_dir', type=str, default='seq_data',
                        help='Directory for saving the processed sequence data.')
    args = vars(parser.parse_args())

    print(args)
    tic = time.time()
    data = data_utils.prepare_data(args, add_self_loop=False, i_1=args['i_1'], i_0=args['i_0'])
    graph2seq(args, data)
    toc = time.time()
    print(f"Elapsed time={toc -tic:.2f}(s)")
