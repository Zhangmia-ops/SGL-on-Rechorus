# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import pandas as pd

from models.BaseModel import GeneralModel


class PaperLightGCN(nn.Module):
    def __init__(self, user_num, item_num, emb_dim, n_layers):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(user_num, emb_dim)
        self.item_emb = nn.Embedding(item_num, emb_dim)

        nn.init.normal_(self.user_emb.weight, std=0.2)
        nn.init.normal_(self.item_emb.weight, std=0.2)

    def forward(self, adj):
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [ego]

        for _ in range(self.n_layers):
            ego = torch.sparse.mm(adj, ego)
            embs.append(ego)

        out = torch.mean(torch.stack(embs, dim=1), dim=1)
        return torch.split(out, [self.user_num, self.item_num])


# ============================================================================
# SGL Model
# ============================================================================
class SGL(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'ssl_temp', 'ssl_reg', 'ssl_ratio', 'aug_type']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64)
        parser.add_argument('--n_layers', type=int, default=2)

        # SSL args
        parser.add_argument('--ssl_temp', type=float, default=0.2)
        parser.add_argument('--ssl_reg', type=float, default=1e-5)
        parser.add_argument('--ssl_ratio', type=float, default=0.1)
        parser.add_argument('--aug_type', type=int, default=1,
                            help='1=edge drop, 2=node drop, 3=random walk')

        return GeneralModel.parse_model_args(parser)

    # ------------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------------
    def __init__(self, args, corpus):
        super().__init__(args, corpus)

        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.ssl_temp = args.ssl_temp
        self.ssl_reg = args.ssl_reg
        self.ssl_ratio = args.ssl_ratio
        self.aug_type = args.aug_type

        # Encoder
        self.encoder = PaperLightGCN(
            self.user_num, self.item_num,
            self.emb_size, self.n_layers
        ).to(self.device)

        # Graph
        self.graph = self._build_graph(corpus)
        self.graph = self.graph.coalesce().to(self.device)

    # ------------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------------
    def _build_graph(self, corpus):
        train_data = corpus.train_clicked_set
        users, items = [], []

        for u, ilist in train_data.items():
            users.extend([u] * len(ilist))
            items.extend(ilist)

        df = pd.DataFrame({'user': users, 'item': items})
        u = df['user'].values
        i = df['item'].values

        n_nodes = self.user_num + self.item_num
        mat = sp.csr_matrix(
            (np.ones_like(u, dtype=np.float32), (u, i + self.user_num)),
            shape=(n_nodes, n_nodes)
        )

        adj = mat + mat.T
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj).dot(d_mat)
        return self._sp_to_tensor(norm_adj)

    def _sp_to_tensor(self, mat):
        coo = mat.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(indices, values, coo.shape)

    # ------------------------------------------------------------------------
    # Graph augmentations
    # ------------------------------------------------------------------------
    def _edge_dropout(self):
        g = self.graph
        idx = g.indices()
        val = g.values()

        mask = torch.rand(len(val), device=val.device) > self.ssl_ratio
        return torch.sparse.FloatTensor(
            idx[:, mask], val[mask], g.shape
        ).coalesce()

    def _node_dropout(self):
        g = self.graph
        idx = g.indices()
        val = g.values()

        node_mask = torch.rand(self.user_num + self.item_num,
                               device=val.device) > self.ssl_ratio
        keep = node_mask[idx[0]] & node_mask[idx[1]]

        return torch.sparse.FloatTensor(
            idx[:, keep], val[keep], g.shape
        ).coalesce()

    def _rw_subgraphs(self):
        v1, v2 = [], []
        for _ in range(self.n_layers):
            v1.append(self._edge_dropout())
            v2.append(self._edge_dropout())
        return v1, v2

    def _rw_forward(self, views):
        ego = torch.cat([
            self.encoder.user_emb.weight,
            self.encoder.item_emb.weight
        ], dim=0)

        embs = [ego]
        for l in range(self.n_layers):
            ego = torch.sparse.mm(views[l], ego)
            embs.append(ego)

        out = torch.mean(torch.stack(embs, dim=1), dim=1)
        return torch.split(out, [self.user_num, self.item_num])

    # ------------------------------------------------------------------------
    # SSL loss (paper-level)
    # ------------------------------------------------------------------------
    def _ssl_loss(self, emb1, emb2, idx):
        z1 = F.normalize(emb1[idx], dim=1)
        z2 = F.normalize(emb2[idx], dim=1)

        logits = torch.matmul(z1, z2.t()) / self.ssl_temp
        labels = torch.arange(len(idx), device=idx.device)
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------------
    def forward(self, feed_dict):
        u_ids = feed_dict['user_id'].long()
        i_ids = feed_dict['item_id'].long()

        u_all, i_all = self.encoder(self.graph)
        pred = (u_all[u_ids][:, None, :] * i_all[i_ids]).sum(-1)

        out = {'prediction': pred.view(feed_dict['batch_size'], -1)}

        # SSL
        if self.training and self.ssl_reg > 0:
            if self.aug_type == 3:
                v1, v2 = self._rw_subgraphs()
                u1, i1 = self._rw_forward(v1)
                u2, i2 = self._rw_forward(v2)
            else:
                g1 = self._edge_dropout() if self.aug_type == 1 else self._node_dropout()
                g2 = self._edge_dropout() if self.aug_type == 1 else self._node_dropout()
                u1, i1 = self.encoder(g1)
                u2, i2 = self.encoder(g2)

            all1 = torch.cat([u1, i1], dim=0)
            all2 = torch.cat([u2, i2], dim=0)
            
            # users: [B]
            user_idx = u_ids

            # items: [B, 2] → [2B]
            item_idx = i_ids.view(-1) + self.user_num

            idx = torch.cat([user_idx, item_idx]).unique()
            ssl = self._ssl_loss(all1, all2, idx)


            out['ssl_loss'] = ssl * self.ssl_reg
        else:
            out['ssl_loss'] = torch.tensor(0.0, device=self.device)

        return out

    # ------------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------------
    def loss(self, out_dict):
        bpr = super().loss(out_dict)
        return bpr + out_dict['ssl_loss']


class SGLImpression(SGL):
    reader = 'ImpressionReader'
    runner = 'ImpressionRunner'
