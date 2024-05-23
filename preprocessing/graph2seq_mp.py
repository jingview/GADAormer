import multiprocessing as mp
import argparse
import os
import numpy as np
import dgl
import torch
from tqdm import tqdm
import time
import data_utils
from joblib import Parallel, delayed


def save_sequence(args, data):
    pass




def graph2seq(pid, args, st, ed, nids, graph_data, sequence_array):
    graph_data = data_utils.prepare_data(args, add_self_loop=False, i_1=args['i_1'], i_0=args['i_0'])

    seq_loader = data_utils.GroupFeatureSequenceLoader(graph_data, fanouts=args['fanouts'],
                                                       grp_norm=args['grp_norm'])
    nids = torch.from_numpy(nids)
    seq_feat = seq_loader.load_batch(nids, pid=pid)
    sequence_array[st:ed] = seq_feat.numpy()



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

    parser.add_argument('--fanouts', type=int, default=[-1], nargs='+',
                        help='Sampling neighbors, default [-1] means full neighbors')

    parser.add_argument('--base_dir', type=str, default='~/.dgl',
                        help='Directory for loading graph data.')
    parser.add_argument('--save_dir', type=str, default='mp_output',
                        help='Directory for saving the processed sequence data.')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Using n processes.')
    parser.add_argument('--i_1', type=int, default=0, help='Number of additional label 1 nodes.')
    parser.add_argument('--i_0', type=int, default=0, help='Number of additional label 0 nodes.')

    args = vars(parser.parse_args())
    print(args)


    graph_data = data_utils.prepare_data(args, add_self_loop=False, i_1=args['i_1'], i_0=args['i_0'])

    g = graph_data.graph
    n_classes = graph_data.n_classes
    feat_dim = graph_data.feat_dim
    n_relations = graph_data.n_relations
    n_groups = n_classes + 1
    n_hops = len(args['fanouts'])
    n_nodes = g.num_nodes()

    seq_len = n_relations * (n_hops * n_groups + 1)

    all_nid = g.nodes()
    file_dir = os.path.join(args['save_dir'], args['dataset'])
    os.makedirs(file_dir, exist_ok=True)

    flag_1 = 'grp_norm' if args['grp_norm'] else 'no_grp_norm'
    flag_2 = 'norm_feat' if args['norm_feat'] else 'no_norm_feat'
#     flag_3 = 'self_loop' if args['add_self_loop'] elsr 'no_self_loop'
    file_name = f"{args['dataset']}_{flag_1}_{flag_2}_{n_hops}_" \
                f"{args['train_size']}_{args['val_size']}_{args['seed']}.npy"
    seq_file = os.path.join(file_dir, file_name)

    sequence_array = np.memmap(seq_file, dtype=np.float32, mode='w+', shape=(n_nodes, seq_len, feat_dim))




    procs = []
    n_workers = args['n_workers']

    nids = g.nodes().numpy()
    block_size = nids.shape[0] // n_workers + 1

    tic = time.time()


    Parallel(n_jobs=n_workers)(
        delayed(graph2seq)(
            pid,
            args,
            pid * block_size,
            min((pid + 1) * block_size, n_nodes),
            nids[pid * block_size: min((pid + 1) * block_size, n_nodes)],
            graph_data,
            sequence_array
        )
        for pid in range(n_workers)

    )

    sequence_array.flush()
    
    toc = time.time()
    print(f"Elapsed Tiem = {toc -tic:.2f}(s)")
