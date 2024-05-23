import torch
import torch.nn as nn


class CustomEncoder(nn.Module):
    def __init__(self, feat_dim, emb_dim, n_classes, n_hops, n_relations, dropout=0.1):

        super(CustomEncoder, self).__init__()
        self.hop_embedding = HopEmbedding(n_hops + 1, emb_dim)
        self.relation_embedding = RelationEmbedding(n_relations, emb_dim)
        self.group_embedding = GroupEmbedding(n_classes + 1, emb_dim)

        # linear  projection
        self.MLP = nn.Sequential(nn.Linear(feat_dim, emb_dim),
                                 nn.ReLU())

        self.dropout = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.emb_dim = emb_dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_classes = n_classes

        self.n_groups = n_classes + 1

        self.base_seq_len = n_hops * (n_classes + 1) + 1

    def forward(self, x):

        device = x.device

        hop_idx = torch.arange(self.n_hops + 1, dtype=torch.int64).to(device)
        rel_idx = torch.arange(self.n_relations, dtype=torch.int64).to(device)
        grp_idx = torch.arange(self.n_groups, dtype=torch.int64).to(device)

        hop_emb = self.hop_embedding(hop_idx)

        center_hop_emb = hop_emb[0].unsqueeze(0)

        hop_emb_list = [center_hop_emb]
        for i in range(1, self.n_hops + 1):
            hop_emb_list.append(hop_emb[i].repeat(self.n_groups, 1))

        hop_emb = torch.cat(hop_emb_list, dim=0).repeat(self.n_relations, 1)

        rel_emb = self.relation_embedding(rel_idx)

        rel_emb = rel_emb.repeat(1, self.base_seq_len).view(-1, self.emb_dim)
        
        grp_emb = self.group_embedding(grp_idx)

        center_grp_emb = grp_emb[-1].unsqueeze(0)

        hop_grp_emb = grp_emb.repeat(self.n_hops, 1)

        grp_emb = torch.cat((center_grp_emb, hop_grp_emb), dim=0).repeat(self.n_relations, 1)

        out = self.MLP(x)

        out = out + hop_emb.unsqueeze(1) + rel_emb.unsqueeze(1) + grp_emb.unsqueeze(1)

        out = self.dropout(out)

        return out


class HopEmbedding(nn.Embedding):
    def __init__(self, max_len, emb_dim=128):

        super(HopEmbedding, self).__init__(max_len, emb_dim)


class RelationEmbedding(nn.Embedding):
    def __init__(self, max_len: int, emb_dim=128):

        super(RelationEmbedding, self).__init__(max_len, emb_dim)


class GroupEmbedding(nn.Embedding):
    def __init__(self, max_len: int, emb_dim=128):

        super(GroupEmbedding, self).__init__(max_len, emb_dim)
