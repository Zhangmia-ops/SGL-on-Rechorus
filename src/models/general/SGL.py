import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import pandas as pd

from models.BaseModel import GeneralModel

class SGL(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    # 将关键参数加入日志
    extra_log_args = ['emb_size', 'n_layers', 'ssl_temp', 'ssl_reg', 'ssl_ratio']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64)
        parser.add_argument('--n_layers', type=int, default=2)
        parser.add_argument('--keep_prob', type=float, default=0.6)
        
        # SSL args
        parser.add_argument('--ssl_temp', type=float, default=0.2, help='Temperature for contrastive loss')
        parser.add_argument('--ssl_reg', type=float, default=1e-5, help='Weight for SSL loss')
        parser.add_argument('--aug_type', type=int, default=1, help='1=edge drop, 2=node drop, 3=rw')
        parser.add_argument('--ssl_ratio', type=float, default=0, help='Ratio for augmentation dropout')
        
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.ssl_temp = args.ssl_temp
        self.ssl_reg = args.ssl_reg
        self.aug_type = args.aug_type
        self.ssl_ratio = args.ssl_ratio
        
        # 记录 L2 参数供参考，但正则化交由 BaseRunner 的 Optimizer 处理
        self.l2 = args.l2 

        # 1. 初始化 Embedding
        self._init_weights()
        
        # 2. 构建图
        self.graph = None
        if hasattr(corpus, 'train_clicked_set'):
            train_data = corpus.train_clicked_set
        elif hasattr(corpus, 'train'):
            train_data = corpus.train
        else:
            train_data = corpus
        self.prepare_graph(train_data)

    def _init_weights(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        nn.init.xavier_normal_(self.u_embeddings.weight)
        nn.init.xavier_normal_(self.i_embeddings.weight)

    def prepare_graph(self, train_data):
        if isinstance(train_data, dict):
            users, items = [], []
            for u, i_list in train_data.items():
                users.extend([u] * len(i_list))
                items.extend(i_list)
            df = pd.DataFrame({'user_id': users, 'item_id': items})
        elif isinstance(train_data, list):
            df = pd.DataFrame(train_data)
        else:
            df = train_data

        user_np = df['user_id'].values
        item_np = df['item_id'].values
        n_users, n_items = self.user_num, self.item_num

        ratings = np.ones_like(user_np, dtype=np.float32)
        mat = sp.csr_matrix(
            (ratings, (user_np, item_np + n_users)), 
            shape=(n_users + n_items, n_users + n_items)
        )
        
        adj = mat + mat.T
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj).dot(d_mat)
        
        self.graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.graph = self.graph.coalesce().to(self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(indices, values, coo.shape)

    def _propagate(self, graph):
        all_emb = torch.cat([self.u_embeddings.weight, self.i_embeddings.weight], dim=0)
        embs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
        
        users, items = torch.split(final_emb, [self.user_num, self.item_num])
        return users, items

    def _graph_dropout(self, graph, keep_prob):
        size = graph.size()
        index = graph.indices().t()
        values = graph.values()

        random_tensor = torch.rand(values.size(), device=values.device) + keep_prob
        mask = random_tensor.floor().bool()
        
        index = index[mask]
        values = values[mask] / keep_prob
        return torch.sparse.FloatTensor(index.t(), values, size)

    def _calc_ssl_loss(self, view1_emb, view2_emb, idx):
        view1 = F.normalize(view1_emb[idx], dim=1)
        view2 = F.normalize(view2_emb[idx], dim=1)
        
        pos_score = (view1 * view2).sum(dim=1)
        ttl_score = torch.matmul(view1, view2.t())
        
        pos_score = torch.exp(pos_score / self.ssl_temp)
        ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
        
        loss = -torch.log(pos_score / ttl_score).mean()
        return loss

    # ----------------------------------------------------------------------------
    # FORWARD
    # ----------------------------------------------------------------------------
    def forward(self, feed_dict):
        u_all, i_all = self._propagate(self.graph)
        
        u_ids = feed_dict['user_id']
        i_ids = feed_dict['item_id']
        
        u_emb = u_all[u_ids]
        i_emb = i_all[i_ids]
        
        prediction = (u_emb[:, None, :] * i_emb).sum(dim=-1)
        
        out_dict = {'prediction': prediction.view(feed_dict['batch_size'], -1)}

        # 训练时计算 SSL Loss
        if self.training and self.ssl_reg > 1e-6:
            # === SSL Loss ===
            # 生成增强视图
            g1 = self._graph_dropout(self.graph, 1 - self.ssl_ratio)
            g2 = self._graph_dropout(self.graph, 1 - self.ssl_ratio)
            
            user_view1, item_view1 = self._propagate(g1)
            user_view2, item_view2 = self._propagate(g2)
            
            # User SSL
            ssl_user = self._calc_ssl_loss(
                torch.cat([user_view1, item_view1], dim=0),
                torch.cat([user_view2, item_view2], dim=0),
                u_ids
            )
            
            # Item SSL (涉及当前 batch 的所有 item，包括负样本)
            flat_items = i_ids.flatten().unique()
            ssl_item = self._calc_ssl_loss(
                torch.cat([user_view1, item_view1], dim=0),
                torch.cat([user_view2, item_view2], dim=0),
                flat_items + self.user_num
            )
            
            out_dict['ssl_loss'] = ssl_user + ssl_item
        else:
            out_dict['ssl_loss'] = torch.tensor(0.0, device=self.device)

        return out_dict

    # ----------------------------------------------------------------------------
    # LOSS
    # ----------------------------------------------------------------------------
    def loss(self, out_dict):
        # 1. BPR Loss (调用父类，基于 'prediction' 计算)
        bpr_loss = super().loss(out_dict)

        # 2. SSL Loss
        ssl_loss = out_dict['ssl_loss'] * self.ssl_reg
        
        # 3. L2 Loss 不需要手动加！ReChorus 的 BaseRunner 会在 optimizer 里做 weight_decay
        
        return bpr_loss + ssl_loss

class SGLImpression(SGL):
    reader = 'ImpressionReader'
    runner = 'ImpressionRunner'

