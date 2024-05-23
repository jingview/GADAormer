import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from modules import embedding
from scipy.io import loadmat
from data import sequence



class TransformerEncoderNet(nn.Module):
    def __init__(self, feat_dim, emb_dim, n_classes, n_hops, n_relations,
                 n_heads, dim_feedforward, n_layers, dropout=0.1, agg_type='attention',seed=114514):

        super(TransformerEncoderNet, self).__init__()


        self.feat_encoder = embedding.CustomEncoder(feat_dim=feat_dim,
                                                    emb_dim=emb_dim, n_relations=n_relations,
                                                    n_hops=n_hops, dropout=dropout,
                                                    n_classes=n_classes)


        encoder_layers = nn.TransformerEncoderLayer(emb_dim, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)


        self.relation_encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(emb_dim, n_heads, dim_feedforward, dropout),
                n_layers
            ) for _ in range(n_relations)
        ])
        self.seed = seed

        if agg_type == 'cat':
            proj_emb_dim = emb_dim * n_relations
        elif agg_type == 'mean':
            proj_emb_dim = emb_dim
        elif agg_type == 'attention':

            self.attn_weight_vectors = nn.Parameter(torch.Tensor(n_relations, emb_dim, emb_dim))
            nn.init.xavier_uniform_(self.attn_weight_vectors.data, gain=1.414)
            proj_emb_dim = emb_dim*2

        self.projection = nn.Sequential(nn.Linear(proj_emb_dim, n_classes))
        self.dropout = nn.Dropout(dropout)

        self.emb_dim = emb_dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_classes = n_classes
        self.agg_type = agg_type

        self.dimension_adjustment_layer = nn.Linear(feat_dim, emb_dim)

        self.init_weights()

        self.features = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.calculate_initial_relation_features()

    def defaultdict_to_sparse_matrix(self, adj_dict):

        num_nodes = max(adj_dict.keys()) + 1 # 获取节点数
        row_indices = []
        col_indices = []
        data = []
        for node, neighbors in adj_dict.items():
            for neighbor in neighbors:
                row_indices.append(node)
                col_indices.append(neighbor)
                data.append(1)
        return csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

    def aug_normalized_adjacency(self, adj):

        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = csr_matrix(np.diag(d_inv_sqrt))
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def calculate_initial_relation_features(self):
        if self.features is None:
            #mat = loadmat('Amazon.mat')
            #features_array = mat['features'].toarray()

            # 用于 '.npy' 文件的路径和数据形状
            filename = './preprocessing/seq_data/yelp/yelp_no_grp_norm_norm_feat_2_0.4_0.1_717.npy'
            dtype = np.float32
            n_nodes, seq_len, d = 45954, 33, 32  # 根据实际数据修改

            # # 使用 memmap 加载 '.npy' 文件
            # memmap_file = np.memmap(filename, dtype=dtype, mode='r', shape=(n_nodes, d))
            # print(memmap_file.shape)
            #
            # # 转换为 PyTorch 张量并移动到相应设备
            # self.features = torch.tensor(memmap_file, dtype=torch.float32).to(self.device)
            # 使用 memmap 加载 '.npy' 文件
            memmap_file = np.memmap(filename, dtype=np.float32, mode='r', shape=(n_nodes, seq_len, d))

            features_2d = np.mean(memmap_file, axis=1)

            self.features = torch.from_numpy(features_2d.copy()).to(self.device)

        self.precomputed_features_list = []
        for adj_file in ['yelp_rur_adjlists.pickle', 'yelp_rtr_adjlists.pickle', 'yelp_rsr_adjlists.pickle']:
            with open(adj_file, 'rb') as f:
                adj_dict = pickle.load(f)
            adj_sparse = self.defaultdict_to_sparse_matrix(adj_dict)
            adj_normalized = self.aug_normalized_adjacency(adj_sparse)
            adj_tensor = torch.FloatTensor(adj_normalized.todense()).to(self.device)
            adj_tensor = torch.mm(adj_tensor, adj_tensor)
            precomputed_features = torch.mm(adj_tensor, self.features)
            self.precomputed_features_list.append(precomputed_features)


    def calculate_batch_relation_features(self, batch_indices):

        concatenated_features = torch.cat(self.precomputed_features_list, dim=0)

        batch_features = concatenated_features[batch_indices]

        transformed_features = self.feat_encoder.MLP(batch_features)


        return transformed_features.to(self.device)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def gat_attention(self, mr_feats):

        attn_weight_vectors = self.attn_weight_vectors


        relation_agg_feats = []

        for r in range(mr_feats.shape[0]):

            attn_scores = torch.matmul(mr_feats[r], attn_weight_vectors[r])
            attn_scores = F.leaky_relu(attn_scores)
            attn_coefficients = torch.softmax(attn_scores, dim=1)


            agg_feats_r = torch.einsum('ni,nj->nj', attn_coefficients, mr_feats[r])
            relation_agg_feats.append(agg_feats_r)

        stacked_feats = torch.stack(relation_agg_feats, dim=0)


        agg_feats = torch.mean(stacked_feats, dim=0)

        return agg_feats

    def cross_relation_agg(self, out):

        device = out.device
        n_tokens = out.shape[0]


        block_len = 1 + self.n_hops * (self.n_classes + 1)
        indices = torch.arange(0, n_tokens, block_len, dtype=torch.int64).to(device)

        mr_feats = torch.index_select(out, dim=0, index=indices)

        relation_feats = []

        for r in range(self.n_relations):
            relation_out = self.relation_encoders[r](mr_feats[r].unsqueeze(0))
            relation_feats.append(relation_out.squeeze(0))

        if self.agg_type == 'cat':

            mr_feats = torch.split(mr_feats, 1, dim=0)


            agg_feats = torch.cat(mr_feats, dim=2).squeeze()

        elif self.agg_type == 'mean':

            agg_feats = torch.mean(mr_feats, dim=0)

        elif self.agg_type =='attention':

            agg_feats = self.gat_attention(mr_feats)

        return agg_feats

    def forward(self, src_emb,batch_indices,src_mask=None):

        src_emb = torch.transpose(src_emb, 1, 0)

        out = self.feat_encoder(src_emb)

        out = self.transformer_encoder(out, src_mask)

        out = self.cross_relation_agg(out)

        current_batch_relation_features = self.calculate_batch_relation_features(batch_indices)

        current_batch_relation_features = current_batch_relation_features.to(self.device)

        out = torch.cat([out, current_batch_relation_features], dim=1)

        out = self.projection(out)


        return out


