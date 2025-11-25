# -*- coding: utf-8 -*-
# 模型封装：复用 HTGNN（保持原实现机制），在外层增加无特征节点的可训练嵌入 + 二分类头
# 说明：仅添加注释与文档字符串，不改动任何业务逻辑

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import math
from dgl.nn.pytorch import GATConv

# --------------------
# 以下为 HTGNN 原版核心模块（RelationAgg / TemporalAgg / HTGNNLayer / HTGNN），
# 保持原逻辑不变（仅格式化注释），便于在本任务中直接复用。
# --------------------

class RelationAgg(nn.Module):
    """跨关系聚合：用可学习权重对不同关系的表示做加权合成。"""
    def __init__(self, n_inp: int, n_hid: int):
        super(RelationAgg, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(n_inp, n_hid),
            nn.Tanh(),
            nn.Linear(n_hid, 1, bias=False)
        )
    def forward(self, h):
        # h: [N, R, D]，先对每个关系投影出权重，再 softmax 归一化
        w = self.project(h).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((h.shape[0],) + beta.shape)
        return (beta * h).sum(1)

class TemporalAgg(nn.Module):
    """时间维聚合：注意力机制 + 位置编码。"""
    def __init__(self, n_inp: int, n_hid: int, time_window: int, device: torch.device):
        super(TemporalAgg, self).__init__()
        self.proj = nn.Linear(n_inp, n_hid)
        self.q_w  = nn.Linear(n_hid, n_hid, bias=False)
        self.k_w  = nn.Linear(n_hid, n_hid, bias=False)
        self.v_w  = nn.Linear(n_hid, n_hid, bias=False)
        self.fc   = nn.Linear(n_hid, n_hid)
        self.pe   = torch.tensor(self.generate_positional_encoding(n_hid, time_window)).float().to(device)
    def generate_positional_encoding(self, d_model, max_len):
        # 标准正余弦位置编码
        pe = np.zeros((max_len, d_model))
        for i in range(max_len):
            for k in range(0, d_model, 2):
                div_term = math.exp(k * - math.log(100000.0) / d_model)
                pe[i][k] = math.sin((i + 1) * div_term)
                if k + 1 < d_model:
                    pe[i][k + 1] = math.cos((i + 1) * div_term)
        return pe
    def forward(self, x):
        # x: [T, N, D] -> 注意力聚合 -> [T, N, D]
        x = x.permute(1, 0, 2)
        h = self.proj(x)
        h = h + self.pe
        q = self.q_w(h); k = self.k_w(h); v = self.v_w(h)
        qk = torch.matmul(q, k.permute(0, 2, 1))
        score = F.softmax(qk, dim=-1)
        h_ = torch.matmul(score, v)
        h_ = F.relu(self.fc(h_))
        return h_

class HTGNNLayer(nn.Module):
    """
    单层 HTGNN：包含
      - intra（同关系、同时间片）基于 GAT 的聚合
      - inter（跨关系）基于 RelationAgg 的聚合
      - across-time（跨时间）基于 TemporalAgg 的聚合
      - 残差 + 可选 LayerNorm
    """
    def __init__(self, graph: dgl.DGLGraph, n_inp: int, n_hid: int, n_heads: int, timeframe: list, norm: bool, device: torch.device, dropout: float):
        super(HTGNNLayer, self).__init__()
        self.n_inp, self.n_hid, self.n_heads = n_inp, n_hid, n_heads
        self.timeframe, self.norm, self.dropout = timeframe, norm, dropout

        # 不同 canonical etype 使用独立的 GATConv
        self.intra_rel_agg = nn.ModuleDict({
            etype: GATConv(n_inp, n_hid, n_heads, feat_drop=dropout, allow_zero_in_degree=True)
            for srctype, etype, dsttype in graph.canonical_etypes
        })
        # 跨关系聚合器（每个时间片一套）
        self.inter_rel_agg = nn.ModuleDict({ttype: RelationAgg(n_hid, n_hid) for ttype in timeframe})
        # 跨时间聚合器（每个节点类型一套）
        self.cross_time_agg = nn.ModuleDict({ntype: TemporalAgg(n_hid, n_hid, len(timeframe), device) for ntype in graph.ntypes})
        # 残差映射与门控
        self.res_fc = nn.ModuleDict({ntype: nn.Linear(n_inp, n_heads * n_hid) for ntype in graph.ntypes})
        self.res_weight = nn.ParameterDict({ntype: nn.Parameter(torch.randn(1)) for ntype in graph.ntypes})
        if norm:
            self.norm_layer = nn.ModuleDict({ntype: nn.LayerNorm(n_hid) for ntype in graph.ntypes})
        self.reset_parameters()
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for ntype in self.res_fc:
            nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)
    def forward(self, graph: dgl.DGLGraph, node_features: dict):
        # intra: 同关系、同时间片（对每个 etype 做消息传递）
        intra_features = dict({ttype:{} for ttype in self.timeframe})
        for stype, etype, dtype in graph.canonical_etypes:
            rel_graph = graph[stype, etype, dtype]
            ttype = etype.split('_')[-1]  # etype 的时间后缀
            dst_feat = self.intra_rel_agg[etype](rel_graph, (node_features[stype][ttype], node_features[dtype][ttype]))
            intra_features[ttype][(stype, etype, dtype)] = dst_feat.squeeze()

        # inter: 跨关系聚合
        inter_features = dict({ntype:{} for ntype in graph.ntypes})
        for ttype in intra_features.keys():
            for ntype in graph.ntypes:
                types_features = []
                for stype, etype, dtype in intra_features[ttype]:
                    if ntype == dtype:
                        types_features.append(intra_features[ttype][(stype, etype, dtype)])
                if len(types_features) == 0:
                    continue
                types_features = torch.stack(types_features, dim=1)
                inter_features[ntype][ttype] = self.inter_rel_agg[ttype](types_features)

        # across-time: 按时间顺序堆叠并做时间注意力
        output_features = {}
        for ntype in inter_features:
            output_features[ntype] = {}
            out_emb = [inter_features[ntype][ttype] for ttype in inter_features[ntype]]
            time_embeddings = torch.stack(out_emb, dim=0)
            h = self.cross_time_agg[ntype](time_embeddings).permute(1,0,2)
            output_features[ntype] = {ttype: h[i] for (i, ttype) in enumerate(self.timeframe)}

        # 残差门控 + (可选)LayerNorm
        new_features = {}
        for ntype in output_features:
            new_features[ntype] = {}
            alpha = torch.sigmoid(self.res_weight[ntype])  # 门控系数 ∈ (0,1)
            for ttype in self.timeframe:
                new_features[ntype][ttype] = output_features[ntype][ttype] * alpha + self.res_fc[ntype](node_features[ntype][ttype]) * (1 - alpha)
                if self.norm:
                    new_features[ntype][ttype] = self.norm_layer[ntype](new_features[ntype][ttype])
        return new_features

class HTGNN(nn.Module):
    """堆叠多层 HTGNNLayer，并把每个时间片的表示求和得到最终节点表示。"""
    def __init__(self, graph: dgl.DGLGraph, n_inp: int, n_hid: int, n_layers: int, n_heads: int, time_window: int, norm: bool, device: torch.device, dropout: float = 0.2):
        super(HTGNN, self).__init__()
        self.n_inp, self.n_hid, self.n_layers, self.n_heads = n_inp, n_hid, n_layers, n_heads
        self.timeframe = [f't{_}' for _ in range(time_window)]
        self.adaption_layer = nn.ModuleDict({ntype: nn.Linear(n_inp, n_hid) for ntype in graph.ntypes})
        self.gnn_layers = nn.ModuleList([HTGNNLayer(graph, n_hid, n_hid, n_heads, self.timeframe, norm, device, dropout) for _ in range(n_layers)])
    def forward(self, graph: dgl.DGLGraph, predict_type: str):
        # 类型特定的线性适配
        inp_feat = {}
        for ntype in graph.ntypes:
            inp_feat[ntype] = {}
            for ttype in self.timeframe:
                inp_feat[ntype][ttype] = self.adaption_layer[ntype](graph.nodes[ntype].data[ttype])
        # 堆叠 GNN 层
        for i in range(self.n_layers):
            inp_feat = self.gnn_layers[i](graph, inp_feat)
        # 目标类型的所有时间片表示求和
        out_feat = sum([inp_feat[predict_type][ttype] for ttype in self.timeframe])
        return out_feat

# --------------------
# 本任务外层封装：给无特征节点（GO / reaction）加 nn.Embedding；二分类头输出 1 维 logits
# --------------------
class ProteinBinaryClassifier(nn.Module):
    """
    基于 HTGNN 的蛋白二分类模型：
      - 对非 protein 节点（如 GO、reaction）注入可训练嵌入（在所有时间片覆盖占位特征）
      - 调用 HTGNN 得到 protein 表示
      - 通过两层 MLP 输出 1 维 logits
    """
    def __init__(self, graph: dgl.DGLHeteroGraph, n_inp: int, d_model: int, n_layers: int, n_heads: int,
                 time_window: int, dropout: float, device: torch.device, use_layernorm: bool = True):
        super().__init__()
        self.device = device
        self.ntypes = list(graph.ntypes)
        # 需要注入嵌入的类型（非 protein）
        self.ntype_need_embed = [nt for nt in self.ntypes if nt != "protein"]
        self.embeds = nn.ModuleDict({nt: nn.Embedding(graph.num_nodes(nt), n_inp) for nt in self.ntype_need_embed})
        # HTGNN 主体
        self.backbone = HTGNN(graph, n_inp=n_inp, n_hid=d_model, n_layers=n_layers, n_heads=n_heads,
                              time_window=time_window, norm=use_layernorm, device=device, dropout=dropout)
        # 二分类头
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    @torch.no_grad()
    def _inject_embeds(self, graph: dgl.DGLHeteroGraph):
        """
        将无特征类型的占位特征，用各自 nn.Embedding 的权重替换（在所有时间片 t0~t35）。
        注：此处使用固定 36 个时间片名称，与数据预处理保持一致。
        """
        for nt in self.ntype_need_embed:
            w = self.embeds[nt].weight  # (N_nt, n_inp)
            # 自动识别所有时间片特征键（以 't' 开头）
            for tname in [k for k in graph.nodes[nt].data.keys() if isinstance(k, str) and k.startswith("t")]:
                graph.nodes[nt].data[tname] = w

    def forward(self, graph: dgl.DGLHeteroGraph):
        # 先覆盖无特征节点的占位特征
        self._inject_embeds(graph)
        # 取 protein 的时序聚合表示
        p_repr = self.backbone(graph, predict_type="protein")   # (N_protein, d_model)
        # 输出 logits（未过 Sigmoid）
        logit = self.cls(p_repr).squeeze(-1)                    # (N_protein,)
        return logit
