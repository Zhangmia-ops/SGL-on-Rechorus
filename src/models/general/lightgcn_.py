# -*- coding: UTF-8 -*-
"""
LightGCN model adapted for BPRMF project environment
Reference:
    "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
    He et al., SIGIR'2020.
"""

import torch
import torch.nn as nn
import numpy as np

from models.BaseModel import GeneralModel

class LightGCN(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'keep_prob']
    
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                           help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=3,
                           help='Number of LightGCN layers.')
        parser.add_argument('--keep_prob', type=float, default=0.6,
                           help='Keep probability for dropout.')
        parser.add_argument('--A_split', type=bool, default=False,
                           help='Whether to split adjacency matrix.')
        parser.add_argument('--dropout', type=bool, default=True,
                           help='Whether to use dropout.')
        return GeneralModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.keep_prob = args.keep_prob
        self.A_split = args.A_split
        self.dropout = args.dropout
        
        # Initialize GeneralModel first to set user_num and item_num
        GeneralModel.__init__(self, args, corpus)
        
        # Initialize model parameters
        self._init_weights()
        self._build_graph()
        
        self.apply(self.init_weights)
    
    def _init_weights(self):
        """Initialize user and item embeddings"""
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        
        # Use normal initialization as in original LightGCN
        nn.init.normal_(self.u_embeddings.weight, std=0.1)
        nn.init.normal_(self.i_embeddings.weight, std=0.1)
    
    def _build_graph(self):
        """Build the graph structure from user-item interactions"""
        # This is a simplified version - in practice you might want to 
        # implement the full sparse graph construction from the original LightGCN
        # For now, we'll create a placeholder that will be computed during training
        self.graph = None
    
    def _convert_sparse_mat_to_tensor(self, X):
        """Convert sparse matrix to sparse tensor"""
        coo = X.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(indices, values, coo.shape)
    
    def __dropout_x(self, x, keep_prob):
        """Dropout for sparse matrix"""
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        """Apply dropout to graph"""
        if self.A_split:
            graph = []
            for g in self.graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.graph, keep_prob)
        return graph
    
    def compute_embeddings(self):
        """
        LightGCN propagation to compute final embeddings
        """
        if self.graph is None:
            # Return initial embeddings if graph is not built
            return self.u_embeddings.weight, self.i_embeddings.weight
        
        all_emb = torch.cat([self.u_embeddings.weight, self.i_embeddings.weight])
        embs = [all_emb]
        
        # Apply dropout during training
        if self.dropout and self.training:
            g_dropped = self.__dropout(self.keep_prob)
        else:
            g_dropped = self.graph
        
        # LightGCN propagation
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_dropped)):
                    temp_emb.append(torch.sparse.mm(g_dropped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_dropped, all_emb)
            embs.append(all_emb)
        
        # Combine embeddings from all layers
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        users, items = torch.split(light_out, [self.user_num, self.item_num])
        return users, items
    
    def forward(self, feed_dict):
        """
        Forward pass
        """
        self.check_list = []
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        
        # Compute final embeddings through LightGCN propagation
        all_users, all_items = self.compute_embeddings()
        
        # Get user and item embeddings for the batch
        cf_u_vectors = all_users[u_ids]
        cf_i_vectors = all_items[i_ids]
        
        # Compute predictions
        prediction = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(dim=-1)
        
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
    
    def prepare_graph(self, interactions):
        """
        Prepare the graph structure from user-item interactions
        This should be called during data preparation
        """
        import scipy.sparse as sp
        
        n_users, n_items = self.user_num, self.item_num
        user_np = interactions['user_id'].numpy()
        item_np = interactions['item_id'].numpy()
        
        # Create adjacency matrix
        ratings = np.ones_like(user_np)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + n_users)), 
                               shape=(n_users + n_items, n_users + n_items))
        
        # Normalize adjacency matrix
        rowsum = np.array(tmp_adj.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        norm_adj = d_mat_inv.dot(tmp_adj).dot(d_mat_inv)
        
        # Convert to tensor
        self.graph = self._convert_sparse_mat_to_tensor(norm_adj).to(self.u_embeddings.weight.device)


class LightGCNImpression(LightGCN):
    reader = 'ImpressionReader'
    runner = 'ImpressionRunner'
    
    @staticmethod
    def parse_model_args(parser):
        return LightGCN.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        super().__init__(args, corpus)
    
    def forward(self, feed_dict):
        out_dict = super().forward(feed_dict)
        
        # For impression version, also return user and item vectors
        u_ids = feed_dict['user_id']
        i_ids = feed_dict['item_id']
        
        all_users, all_items = self.compute_embeddings()
        cf_u_vectors = all_users[u_ids]
        cf_i_vectors = all_items[i_ids]
        
        u_v = cf_u_vectors.repeat(1, i_ids.shape[1]).view(i_ids.shape[0], i_ids.shape[1], -1)
        i_v = cf_i_vectors
        
        out_dict.update({'u_v': u_v, 'i_v': i_v})
        return out_dict