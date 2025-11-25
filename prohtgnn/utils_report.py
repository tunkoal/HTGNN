# -*- coding: utf-8 -*-
# 打印构图信息与分割统计 —— 严格对齐你给的格式
# 说明：仅添加注释与文档字符串，不改动任何业务逻辑

def _pct(n, d):
    """安全百分比：分母为 0 时返回 0.0。"""
    return 0.0 if d == 0 else n * 100.0 / d

def print_graph_report(meta):
    """
    终端报表打印（图与划分概览）。

    参数：
      meta(dict) 需要包含：
        - type2num: {ntype: count}                   # 各节点类型数量
        - label_pos, label_neg, protein_total        # 标签统计
        - base_rel_counts: {(src, base, dst): cnt}   # 基础关系（去时间后缀）聚合后的边数量
        - splits: {"train_idx","val_idx","test_idx"} # 三个划分（protein 索引）
        - y: Tensor                                  # protein 标签（与索引一致）
    """
    t2n = meta["type2num"]
    total_nodes = sum(t2n.values())
    pos, neg, p_total = meta["label_pos"], meta["label_neg"], meta["protein_total"]

    # Header
    print("========== Graph Report ==========")
    print(f"[Nodes] total = {total_nodes:,}")
    # 逐类型按预设顺序展示（若存在）
    order = ["GO", "protein", "reaction"]
    for key in order:
        if key in t2n:
            cnt = t2n[key]
            print(f"  - {key:<9}: {cnt:7,} ({_pct(cnt, total_nodes):6.2f}%)")

    print(f"[Labels on protein] total = {p_total:,}")
    print(f"  - positive : {pos:7,} ({_pct(pos, p_total):6.2f}%)")
    print(f"  - negative : {neg:7,} ({_pct(neg, p_total):6.2f}%)\n")

    # 边聚合统计（不带时间后缀）
    total_edges = sum(meta["base_rel_counts"].values())
    print(f"[Edges] total = {total_edges:,}")
    print("  Aggregated by base relation:")
    # 固定展示顺序（如存在）以接近示例
    pref = [
        ("protein","protein-protein","protein"),
        ("protein","co_localized","protein"),
        ("protein","protein-reaction","reaction"),
        ("protein","co_reaction","protein"),
        ("protein","enables","GO"),
        ("protein","located_in","GO"),
        ("protein","involved_in","GO"),
        ("protein","acts_upstream_of_or_within","GO"),
    ]
    for key in pref:
        k = key
        cnt = meta["base_rel_counts"].get(k, 0)
        if cnt > 0:
            print(f"    * ({k[0]:<7}, {k[1]:<27}, {k[2]:<8}) : {cnt:,}")

    # 切分概览与类内比例
    tr, va, te = meta["splits"]["train_idx"], meta["splits"]["val_idx"], meta["splits"]["test_idx"]
    print("[Splits] size and ratio over labeled nodes")
    tr_pos = int(meta['y'][tr].sum().item());
    tr_neg = len(tr) - tr_pos
    va_pos = int(meta['y'][va].sum().item());
    va_neg = len(va) - va_pos
    te_pos = int(meta['y'][te].sum().item());
    te_neg = len(te) - te_pos
    print(
        f"  - train: {len(tr):5,} ({_pct(len(tr), p_total):6.2f}%) | [+] {tr_pos:5,} ({_pct(tr_pos, len(tr)):6.2f}%)  [-] {tr_neg:5,} ({_pct(tr_neg, len(tr)):6.2f}%)")
    print(
        f"  - val  : {len(va):5,} ({_pct(len(va), p_total):6.2f}%) | [+] {va_pos:5,} ({_pct(va_pos, len(va)):6.2f}%)  [-] {va_neg:5,} ({_pct(va_neg, len(va)):6.2f}%)")
    print(
        f"  - test : {len(te):5,} ({_pct(len(te), p_total):6.2f}%) | [+] {te_pos:5,} ({_pct(te_pos, len(te)):6.2f}%)  [-] {te_neg:5,} ({_pct(te_neg, len(te)):6.2f}%)")
    print("==================================")
