import dgl
import copy
import torch
import numpy as np
import math
from collections import namedtuple
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from data import fraud_dataset, data_helper



class GroupFeatureSequenceLoader:
    def __init__(self, graph_data: namedtuple, default_feat=None, fanouts=None, grp_norm=False):

        if not default_feat:
            default_feat = torch.zeros(graph_data.feat_dim)

        self.default_feat = default_feat
        self.relations = list(graph_data.graph.etypes)
        self.features = graph_data.features
        self.labels = graph_data.labels
        self.pseudo_labels = graph_data.pseudo_labels
        self.n_groups = graph_data.n_classes +1
        self.grp_norm = grp_norm
        self.graph = graph_data.graph
        self.fanouts = [-1] if fanouts is None else fanouts

        self.train_nid = graph_data.train_nid
        self.val_nid = graph_data.val_nid
        self.test_nid = graph_data.test_nid
        self.train_nid_set = set(self.train_nid.tolist())
        self.original_train_nid = graph_data.original_train_nid

        self.val_nid_set = set(self.val_nid.tolist())
        self.test_nid_set = set(self.test_nid.tolist())
        self.all_group_sizes = []

    def load_batch(self, batch_nid: torch.Tensor, device='cpu', pid=0):
        grp_feat_list = []

        if batch_nid.shape == torch.Size([]):
            batch_nid = batch_nid.unsqueeze(0)
        cnt = 0
        for nid in tqdm(batch_nid):

            feat_list = []

            neighbor_dict = self._sample_multi_hop_neighbors(nid)


            for etype in self.graph.etypes:
                multi_hop_neighbor_list = neighbor_dict[etype]

                feat_list.append(self._group_aggregation(nid, batch_nid, multi_hop_neighbor_list))

            grp_feat = torch.cat(feat_list, dim=0)
            grp_feat_list.append(grp_feat)


            del neighbor_dict

        batch_feats = torch.stack(grp_feat_list, dim=0)

        return batch_feats

    def _sample_multi_hop_neighbors(self, nid, replace=True, probs=None):

        sampling_results = {}
        for etype in self.graph.etypes:
            rel_g = self.graph.edge_type_subgraph([etype])
            multi_hop_neighbor_list = []
            nbs = nid.item()
            for max_degree in self.fanouts:

                if max_degree == -1:
                    nbs = rel_g.in_edges(nbs)[0]
                else:
                    nbs = rel_g.in_edges(nbs)[0]
                    sample_num = min(nbs.shape[0], max_degree)
                    nbs = np.random.choice(nbs.numpy(), size=(sample_num,), replace=replace, p=probs)
                    nbs = torch.LongTensor(nbs)

                nbs = nbs.unique()
                multi_hop_neighbor_list.append(nbs)

            sampling_results[etype] = multi_hop_neighbor_list

        return sampling_results

    def _kmeans_aggregation(self, multi_hop_neighbors):
        pass

    def _group_aggregation(self, nid, batch_nid, multi_hop_neighbor_list):
        nid = nid.item()
        center_feat = self.features[nid]
        feat_list = [center_feat.unsqueeze(0)]
        group_sizes = torch.empty(0, dtype=torch.float32).to(self.features.device)
        group_sizes_list = [1]

        all_neighbors = set()
        for neighbors in multi_hop_neighbor_list:
            all_neighbors.update(neighbors.tolist())


        for neighbors in multi_hop_neighbor_list:
            if neighbors.shape == torch.Size([0]):
                agg_feat = torch.stack([self.default_feat, self.default_feat, self.default_feat, self.default_feat, self.default_feat], dim=0)
            else:
                nb_set = set(neighbors.tolist())

                batch_nid_set = set(batch_nid.tolist())

                unmasked_set = nb_set.intersection(self.train_nid_set)

                unmasked_set.discard(nid)
                unmasked_nid = torch.LongTensor(list(unmasked_set))


                masked_set = nb_set.difference(unmasked_set)
                masked_nid = torch.LongTensor(list(masked_set))

                pos_nid = unmasked_nid[self.labels[unmasked_nid] == 1]  
                neg_nid = unmasked_nid[self.labels[unmasked_nid] == 0]  
                pseudo_labels_neg_nid = unmasked_nid[self.pseudo_labels[unmasked_nid] == 0]
                pseudo_labels_pos_nid = unmasked_nid[self.pseudo_labels[unmasked_nid] == 1]

                num_nodes_h0 = len(neg_nid)
                num_nodes_h1 = len(pos_nid)
                num_nodes_h2 = len(masked_nid)
                num_nodes_h3 = len(pseudo_labels_neg_nid)
                num_nodes_h4 = len(pseudo_labels_pos_nid)

                h_0 = self._feat_aggregation(neg_nid)
                h_1 = self._feat_aggregation(pos_nid)
                h_2 = self._feat_aggregation(masked_nid)
                h_3 = self._feat_aggregation(pseudo_labels_neg_nid)
                h_4 = self._feat_aggregation(pseudo_labels_pos_nid)


                agg_feat = torch.stack([h_0, h_1, h_2, h_3, h_4], dim=0)

                assert nid not in unmasked_set, f"error, node {nid} label leakage"


            feat_list.append(agg_feat)

        feat_sequence = torch.cat(feat_list, dim=0)

        return feat_sequence



    def _feat_aggregation(self, nids: torch.LongTensor):
        if nids.shape == torch.Size([0]):
            return self.default_feat
        feats = torch.index_select(self.features, dim=0, index=nids)
        feats = torch.mean(feats, dim=0)

        if self.grp_norm is True:
            feats = feats * (1 / math.sqrt(nids.shape[0]))


        return feats


def prepare_data(args, add_self_loop=False, i_1=0, i_0=0):
    g = load_graph(dataset_name=args['dataset'], raw_dir=args['base_dir'],
                   train_size=args['train_size'], val_size=args['val_size'],
                   seed=args['seed'], norm=args['norm_feat'],
                   force_reload=args['force_reload'], i_1=i_1, i_0=i_0)

    relations = list(g.etypes)
    if add_self_loop is True:
        for etype in relations:
            g = dgl.remove_self_loop(g, etype=etype)
            g = dgl.add_self_loop(g, etype=etype)

        print('add self-loop for ', g)

    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    labels = g.ndata['label'].squeeze().long()

    dataset = args['dataset']

    if dataset == 'amazon':
        embeds_file = 'Amazon_embeds.npy'
    elif dataset == 'yelp':
        embeds_file = 'Yelpchi_embeds.npy'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    embeds = np.load(embeds_file)
    embeds = torch.from_numpy(embeds)  

    train_nid = torch.nonzero(train_mask, as_tuple=True)[0]
    val_nid = torch.nonzero(val_mask, as_tuple=True)[0]
    test_nid = torch.nonzero(test_mask, as_tuple=True)[0]

    embeds = embeds.squeeze(0)

    train_embs = embeds[train_nid]
    val_embs = embeds[val_nid]
    test_embs = embeds[test_nid]

    train_lbls = labels[train_nid]

    train_embs_pos = train_embs[train_lbls == 1]
    train_embs_neg = train_embs[train_lbls == 0]
    val_test_embs = torch.cat((val_embs, test_embs), dim=0)

    distance_matrix = torch.cdist(val_test_embs, train_embs)

    nearest_labels = []

    for i in range(distance_matrix.shape[0]):
        nearest_train_idx = torch.argmin(distance_matrix[i])
        nearest_labels.append(labels[train_nid[nearest_train_idx]])

    nearest_labels = torch.tensor(nearest_labels)


    topk_indices_pos = []
    topk_indices_neg = []

    val_test_nid = val_nid.tolist() + test_nid.tolist()

    pos_counter, neg_counter = 0, 0
    for i, lbl in enumerate(nearest_labels):
        if lbl == 1 and pos_counter < i_1:
            topk_indices_pos.append(val_test_nid[i])
            pos_counter += 1
        elif lbl == 0 and neg_counter < i_0:
            topk_indices_neg.append(val_test_nid[i])
            neg_counter += 1

    topk_indices_pos = list(set(topk_indices_pos))
    topk_indices_neg = list(set(topk_indices_neg))

    topk_nearest_indices = topk_indices_pos + topk_indices_neg

    topk_nearest_indices = list(set(topk_nearest_indices))

    additional_train_nid = torch.tensor(topk_nearest_indices, dtype=torch.long)

    pseudo_labels = labels.clone()

    original_train_nid = train_nid.clone()
    additional_train_nid = torch.tensor(topk_nearest_indices, dtype=torch.long)
    train_nid = torch.cat([train_nid, additional_train_nid])
    train_mask = torch.zeros_like(train_mask)
    train_mask[train_nid] = 1

    for idx in topk_nearest_indices:
        if idx not in original_train_nid.tolist():
            pseudo_labels[idx] = labels[idx]  # 或者根据需要设定特定的标签


    specific_node_ids = topk_indices_pos + topk_indices_neg

    val_test_nid_set = set(val_nid.tolist() + test_nid.tolist())

    for idx in topk_indices_pos:
        pseudo_labels[idx] = 1  # 假定这些节点应该有伪标签1
    for idx in topk_indices_neg:
        pseudo_labels[idx] = 0  # 假定这些节点应该有伪标签0




    n_relations = len(g.etypes)
    features = g.ndata['feature']
    feat_dim = features.shape[1]
    labels = g.ndata['label'].squeeze().long()

    n_classes = 4


    g.ndata['pseudo_label'] = pseudo_labels



    print(f"[Global] Dataset <{args['dataset']}> Overview\n"
          f"\tEntire (postive/total) {torch.sum(labels):>6} / {labels.shape[0]:<6}\n"
          f"\tTrain  (postive/total) {torch.sum(labels[original_train_nid]):>6} / {labels[original_train_nid].shape[0]:<6}\n"
          f"\tValid  (postive/total) {torch.sum(labels[val_nid]):>6} / {labels[val_nid].shape[0]:<6}\n"
          f"\tTest   (postive/total) {torch.sum(labels[test_nid]):>6} / {labels[test_nid].shape[0]:<6}\n")

    Datatype = namedtuple('GraphData', ['graph', 'features', 'labels', 'train_nid', 'val_nid',
                                        'test_nid', 'n_classes', 'feat_dim', 'n_relations','pseudo_labels','original_train_nid'])
    graph_data = Datatype(graph=g, features=features, labels=labels,
                          train_nid=train_nid,val_nid=val_nid, test_nid=test_nid, n_classes=n_classes,
                          feat_dim=feat_dim, n_relations=n_relations,pseudo_labels=pseudo_labels,original_train_nid=original_train_nid,)

    return graph_data

def load_graph(dataset_name, train_size, val_size, seed, norm, force_reload, i_1, i_0, raw_dir='~/.dgl/'):

    if dataset_name in ['amazon', 'yelp', 'mimic']:
        fraud_data = fraud_dataset.FraudDataset(dataset_name, train_size=train_size, val_size=val_size,
                                                random_seed=seed, force_reload=force_reload)

    g = fraud_data[0]


    if norm and (dataset_name not in ['BF10M']):
        h = data_helper.row_normalize(g.ndata['feature'], dtype=np.float32)
        g.ndata['feature'] = torch.from_numpy(h)
    else:
        g.ndata['feature'] = g.ndata['feature'].float()

    return g





