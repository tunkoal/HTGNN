# -*- coding: utf-8 -*-
# 读取 CSV，解析 timeset -> 36 个时间片，构建异构时态图所需的中间结构
# 说明：仅添加注释与文档字符串，不改动任何业务逻辑

import pandas as pd
import numpy as np
import re
import torch
import random

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# 36 个时间片名称（t0 ~ t35）
TIME_SLICES = [f"t{i}" for i in range(36)]

def _normalize_node_type(ntype: str) -> str:
    """
    轻量归一化节点类型：
      - "single protein" -> "protein"
      - 其余保持不变
    """
    ntype = (ntype or "").strip()
    return "protein" if ntype == "single protein" else ntype

def load_nodes(nodes_csv: str):
    """
    读取节点表并生成多种映射字典。

    参数：
      nodes_csv: 节点 CSV 路径，至少包含列：
        - Id（节点 ID）
        - node type（节点类型）
        - function（可选；protein 的标签依据此列是否为 ORF-V）

    返回：
      nodes_df  : DataFrame（node type 已做简单归一化）
      id2type   : {node_id(str): node_type(str)}
      type2ids  : {node_type: [node_id,...]}
      labels_p  : {protein_id: 0/1}，仅 protein 有标签（function: ORF-V -> 1）
    """
    df = pd.read_csv(
        nodes_csv,
        dtype={"Id": str, "node type": str, "function": str},  # 关键列固定为字符串
    )
    df["node type"] = df["node type"].map(_normalize_node_type)

    # 为 protein 生成二分类标签（是否 ORF-V）
    labels = {}
    for _, r in df.iterrows():
        if r["node type"] == "protein":
            lab = 1 if str(r.get("function", "")).strip().upper() == "ORF-V" else 0
            labels[str(r["Id"])] = int(lab)

    # 节点 ID -> 节点类型
    id2type = {str(r["Id"]): r["node type"] for _, r in df.iterrows()}

    # 节点类型 -> 节点 ID 列表
    type2ids = {}
    for nid, nt in id2type.items():
        type2ids.setdefault(nt, []).append(nid)

    return df, id2type, type2ids, labels

# 用于 timeset 解析的正则
_FLOAT = r"-?\d+(?:\.\d+)?"
_PAIR  = re.compile(r"\[\s*("+_FLOAT+r")\s*,\s*("+_FLOAT+r")\s*\]")
_NUM   = re.compile(_FLOAT)

def parse_timeset_to_slices(timeset_str: str, boundaries: list, eps: float = 1e-9):
    """
    将 "[a,b],[c,d],..." 解析到被覆盖的时间片索引。
    - 空值 -> 全时段
    - 半开区间 [a,b)
    - 浮点端点允许微小误差（eps）
    """
    T = len(boundaries) - 1  # 时间片数量
    if not timeset_str or (isinstance(timeset_str, float) and np.isnan(timeset_str)):
        return list(range(T))

    spans = _PAIR.findall(str(timeset_str))
    if not spans:
        raise ValueError(f"[timeset] 无法解析: {timeset_str}")

    # 容差下的“端点→下标”映射
    def to_idx(x: float) -> int:
        for i, b in enumerate(boundaries):
            if abs(b - x) <= eps:
                return i
        raise ValueError(f"[timeset] 端点不在边界集中: {x}")

    used = set()
    for a_str, b_str in spans:
        ia = to_idx(float(a_str)); ib = to_idx(float(b_str))
        for t in range(ia, ib):
            if 0 <= t < T:
                used.add(t)
    return sorted(used)


def collect_boundaries_from_edges(edges_df: pd.DataFrame):
    """
    从边表抽取所有 timeset 的端点，得到去重后排序的边界列表。

    要求：
      - 唯一端点数必须为 37（形成 36 个时间片）

    返回：
      uniq: 升序唯一端点（list）
    """
    vals = []
    for s in edges_df.get("timeset", []):
        if isinstance(s, str):
            vals.extend(map(float, _NUM.findall(s)))
    uniq = sorted(set(vals))
    if len(uniq) != 37:
        raise ValueError(f"[timeset] 唯一端点应为 37 个，当前为 {len(uniq)}。")
    return uniq

def load_edges(edges_csv: str, id2type: dict):
    """
    读取边表并按时间片展开。

    参数：
      edges_csv: 边 CSV 路径，需包含列：
        - Source（源节点 ID）
        - Target（目标节点 ID）
        - edge_type（关系类型，不含时间后缀）
        - Type（Directed/Undirected）
        - timeset（形如 "[a,b],[c,d],..."）
      id2type : {node_id: node_type}，用于过滤无效节点

    返回：
      edges_df   : 原始边 DataFrame
      boundaries : 37 个端点（升序）
      edges_per_t: {t: [ (src_id, dst_id, edge_type, directed_bool) , ... ]}
    """
    df = pd.read_csv(
        edges_csv,
        dtype={"Source": str, "Target": str, "edge_type": str, "Type": str, "timeset": str},
    )
    boundaries = collect_boundaries_from_edges(df)
    edges_per_t = {t: [] for t in range(len(boundaries) - 1)}

    for _, r in df.iterrows():
        sid = str(r["Source"]); tid = str(r["Target"])
        et  = str(r["edge_type"]).strip()
        is_dir_str = str(r["Type"]).strip()
        if is_dir_str not in ("Directed", "Undirected"):
            raise ValueError(f"[edges.Type] 只允许 Directed/Undirected，遇到: {is_dir_str}")
        is_dir = (is_dir_str == "Directed")

        # 将 timeset 映射为覆盖的时间片下标
        used_ts = parse_timeset_to_slices(r.get("timeset", ""), boundaries)

        # 节点必须存在于 id2type 映射，否则跳过
        if sid not in id2type or tid not in id2type:
            continue

        # 在对应时间片登记该边
        for t in used_ts:
            edges_per_t[t].append((sid, tid, et, is_dir))
    return df, boundaries, edges_per_t
