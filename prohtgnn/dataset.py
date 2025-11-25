# -*- coding: utf-8 -*-
# 把预处理结果真正构造成 DGL 异构时态图；并产出分层抽样的 train/val/test 划分
# 说明：仅添加注释与文档字符串，不改动任何业务逻辑

import torch
import numpy as np
import random
import dgl
from sklearn.model_selection import StratifiedShuffleSplit
from preprocess import load_nodes, load_edges, TIME_SLICES

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dgl.seed(SEED)

def build_hetero_time_graph(nodes_csv: str, edges_csv: str):
    """
    从 CSV 构建 DGL 异构时态图，并生成数据划分与元信息。

    步骤：
      1) 调用 preprocess 读取节点/边与时间片展开
      2) 为每种节点类型建立“字符串 ID -> 连续索引”的映射
      3) 组装 hetero_graph 的 (srctype, etype_t{i}, dsttype) 到 (u,v) 边列表
      4) 构造 DGLHeteroGraph 与节点特征
      5) 生成 protein 标签 y 和 分层划分 split
      6) 汇总 meta（便于报告/统计）

    返回：
      G, y, split, meta, n_inp
        - G: 异构图（边类型已按时间片展开）
        - y: protein 节点的 0/1 标签（FloatTensor）
        - split: 70/15/15 分层划分（train/val/test 的索引）
        - meta: 统计信息（节点计数、关系聚合计数、标签分布等）
        - n_inp: 输入特征维度（此处为 1）
    """
    print("构建异构时态图 ...")
    # 1) 读取并解析
    nodes_df, id2type, type2ids, labels_p = load_nodes(nodes_csv)
    _, _, edges_per_t = load_edges(edges_csv, id2type)

    # 2) 每种节点类型建立 ID -> 连续索引
    ntype2nid = {}
    for ntype, id_list in type2ids.items():
        ntype2nid[ntype] = {nid: i for i, nid in enumerate(sorted(id_list))}

    # 3) 组装 heterograph 的三元组键：etype 带时间后缀
    hetero_dict = {}
    for t in range(len(TIME_SLICES)):
        for (sid, tid, et, is_dir) in edges_per_t[t]:
            st = id2type[sid]; dt = id2type[tid]
            s_idx = ntype2nid[st][sid]; d_idx = ntype2nid[dt][tid]
            k = (st, f"{et}_t{t}", dt)
            hetero_dict.setdefault(k, ([], []))
            hetero_dict[k][0].append(s_idx); hetero_dict[k][1].append(d_idx)
            # 若是无向边，则补反向一条（仍使用相同的 base etype_t{t}）
            if not is_dir:
                k2 = (dt, f"{et}_t{t}", st)
                hetero_dict.setdefault(k2, ([], []))
                hetero_dict[k2][0].append(d_idx); hetero_dict[k2][1].append(s_idx)

    # 4) 构图（node 数量 dict）
    num_nodes_dict = {ntype: len(nid_map) for ntype, nid_map in ntype2nid.items()}
    G = dgl.heterograph(hetero_dict, num_nodes_dict=num_nodes_dict)

    # 5) 节点特征：
    #    - protein: 从 nodes_df 读取 t0~t35 的标量特征
    #    - GO/reaction: 仅占位（1 维 0），模型里会被 nn.Embedding 覆盖
    n_inp = 1  # 每个时间片一个标量 -> 通过 HTGNN 的适配层做升维
    for ntype in G.ntypes:
        for i, tname in enumerate(TIME_SLICES):
            G.nodes[ntype].data[tname] = torch.zeros(G.num_nodes(ntype), n_inp).float()

    if "protein" in type2ids:
        pids = sorted(type2ids["protein"])
        for col in TIME_SLICES:
            vals = []
            for pid in pids:
                row = nodes_df.loc[nodes_df["Id"] == pid]
                if row.empty:
                    raise ValueError(f"[nodes] 找不到 protein 节点: {pid}")
                v = row.iloc[0].get(col, np.nan)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    raise ValueError(f"[nodes] protein 节点 {pid} 的 {col} 缺失")
                vals.append([float(v)])
            G.nodes["protein"].data[col] = torch.tensor(vals).float()

    # 6) 生成 protein 标签与分层划分（70/15/15）
    protein_ids_sorted = sorted(type2ids.get("protein", []))
    y = torch.tensor([labels_p.get(pid, 0) for pid in protein_ids_sorted]).float()

    idx_all = np.arange(len(protein_ids_sorted))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)  # 先留出 30%
    train_idx, vt_idx = next(sss1.split(idx_all, y.numpy()))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)  # 剩余对半分
    val_idx, test_idx = next(sss2.split(vt_idx, y[vt_idx].numpy()))
    val_idx = vt_idx[val_idx]; test_idx = vt_idx[test_idx]

    split = {
        "train_idx": torch.tensor(train_idx, dtype=torch.long),
        "val_idx"  : torch.tensor(val_idx, dtype=torch.long),
        "test_idx" : torch.tensor(test_idx, dtype=torch.long),
    }

    # 统计边按“基础关系”（不含时间后缀）的聚合数量（跨时间去重，方向保留）
    # 构造跨时间的有序唯一边集合：同一 (sid, tid, et) 在多个时间片只算一次
    uniq_ordered_edges = set()
    for t in range(len(TIME_SLICES)):
        for (sid, tid, et, is_dir) in edges_per_t[t]:
            if is_dir:
                # 有向边：只加入一个方向
                uniq_ordered_edges.add((sid, tid, et))
            else:
                # 无向边：两个有序方向都计数（方向保留）
                uniq_ordered_edges.add((sid, tid, et))
                uniq_ordered_edges.add((tid, sid, et))

    # 基于唯一边集合做按类型-基础关系的计数
    base_rel_counts = {}
    for (sid, tid, et) in uniq_ordered_edges:
        srctype = id2type[sid]
        dsttype = id2type[tid]
        key = (srctype, et, dsttype)  # et 已是不带时间后缀的基础关系名
        base_rel_counts[key] = base_rel_counts.get(key, 0) + 1

    # 8) 元信息打包（供报告/训练使用）
    meta = {
        "type2num": {nt: G.num_nodes(nt) for nt in G.ntypes},
        "base_rel_counts": base_rel_counts,
        "label_pos": int(y.sum().item()),
        "label_neg": int((len(y) - int(y.sum().item()))),
        "protein_total": len(y),
        "splits": {k: v.numpy() for k, v in split.items()},
        "y": y,
        "protein_order": protein_ids_sorted,
    }
    return G, y, split, meta, n_inp
