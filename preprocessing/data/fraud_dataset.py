import os
from scipy import io, sparse
import numpy as np

from dgl.data.utils import save_graphs, load_graphs, _get_dgl_url
from dgl.convert import heterograph
from dgl.data import DGLBuiltinDataset
from dgl import backend as F


class FraudDataset(DGLBuiltinDataset):
    file_urls = {
        'yelp': 'dataset/FraudYelp.zip',
        'amazon': 'dataset/FraudAmazon.zip',
    }
    relations = {
        'yelp': ['net_rsr', 'net_rtr', 'net_rur'],
        'amazon': ['net_upu', 'net_usu', 'net_uvu'],
    }
    file_names = {
        'yelp': 'YelpChi.mat',
        'amazon': 'Amazon.mat',
    }
    node_name = {
        'yelp': 'review',
        'amazon': 'user',
    }

    def __init__(self, name, raw_dir=None, random_seed=717, train_size=0.7,
                 val_size=0.1, force_reload=False, verbose=True):
        assert name in ['yelp', 'amazon'], 
        url = _get_dgl_url(self.file_urls[name])
        self.seed = random_seed
        self.train_size = train_size
        self.val_size = val_size
        super(FraudDataset, self).__init__(name=name,
                                           url=url,
                                           raw_dir=raw_dir,
                                           hash_key=(random_seed, train_size, val_size),
                                           force_reload=force_reload,
                                           verbose=verbose)

    def process(self):
        file_path = os.path.join(self.raw_path, self.file_names[self.name])

        data = io.loadmat(file_path)

        if sparse.issparse(data['features']):
            node_features = data['features'].todense()
        else:
            node_features = data['features']
        node_labels = data['label'].squeeze()

        graph_data = {}
        for relation in self.relations[self.name]:
            adj = data[relation].tocoo()
            row, col = adj.row, adj.col
            graph_data[(self.node_name[self.name], relation, self.node_name[self.name])] = (row, col)
        g = heterograph(graph_data)

        g.ndata['feature'] = F.tensor(node_features, dtype=F.data_type_dict['float32'])
        g.ndata['label'] = F.tensor(node_labels, dtype=F.data_type_dict['int64'])
        self.graph = g

        self._random_split(g.ndata['feature'], self.seed, self.train_size, self.val_size)

    def __getitem__(self, idx):
         assert idx == 0, 
        return self.graph

    def __len__(self):
        return len(self.graph)

    @property
    def num_classes(self):

        return 2

    def save(self):
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph_{}.bin'.format(self.hash))
        save_graphs(str(graph_path), self.graph)

    def load(self):
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph_{}.bin'.format(self.hash))
        graph_list, _ = load_graphs(str(graph_path))
        g = graph_list[0]
        self.graph = g

    def has_cache(self):
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph_{}.bin'.format(self.hash))
        return os.path.exists(graph_path)

    def _random_split(self, x, seed=717, train_size=0.7, val_size=0.1):
        assert 0 <= train_size + val_size <= 1, \


        N = x.shape[0]
        index = np.arange(N)
        #if self.name == 'amazon':
            # 0-3304 are unlabeled nodes
        #    index = np.arange(3305, N)

        index = np.arange(0, N)
        index = np.random.RandomState(seed).permutation(index)
        train_idx = index[:int(train_size * len(index))]
        val_idx = index[len(index) - int(val_size * len(index)):]
        test_idx = index[int(train_size * len(index)):len(index) - int(val_size * len(index))]
        train_mask = np.zeros(N, dtype=np.bool)
        val_mask = np.zeros(N, dtype=np.bool)
        test_mask = np.zeros(N, dtype=np.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        self.graph.ndata['train_mask'] = F.tensor(train_mask)
        self.graph.ndata['val_mask'] = F.tensor(val_mask)
        self.graph.ndata['test_mask'] = F.tensor(test_mask)


class FraudYelpDataset(FraudDataset):
    def __init__(self, raw_dir=None, random_seed=717, train_size=0.7,
                 val_size=0.1, force_reload=False, verbose=True):
        super(FraudYelpDataset, self).__init__(name='yelp',
                                               raw_dir=raw_dir,
                                               random_seed=random_seed,
                                               train_size=train_size,
                                               val_size=val_size,
                                               force_reload=force_reload,
                                               verbose=verbose)


class FraudAmazonDataset(FraudDataset):
     def __init__(self, raw_dir=None, random_seed=717, train_size=0.7,
                 val_size=0.1, force_reload=False, verbose=True):
        super(FraudAmazonDataset, self).__init__(name='amazon',
                                                 raw_dir=raw_dir,
                                                 random_seed=random_seed,
                                                 train_size=train_size,
                                                 val_size=val_size,
                                                 force_reload=force_reload,
                                                 verbose=verbose)


