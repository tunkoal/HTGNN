# -*- coding: utf-8 -*-
# 训练脚本（集中式配置；不使用命令行参数）
# 说明：仅添加注释与文档字符串，不改动任何业务逻辑

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import dgl

from dataset import build_hetero_time_graph
from model import ProteinBinaryClassifier
from utils.pytorchtools import EarlyStopping   # 直接沿用原仓库早停实现
from metrics import compute_all_metrics, plot_lines, plot_roc_pr
from utils_report import print_graph_report

# ========= 配置（集中在顶部） =========
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
NODES_CSV = "data/nodes_with_essential.csv"
EDGES_CSV = "data/edges.csv"

SEED        = 42
DEVICE      = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
TIME_WINDOW = 36
D_MODEL     = 32     # 隐藏维度（与 OGBN-MAG 配置一致）
N_LAYERS    = 2
N_HEADS     = 1
DROPOUT     = 0.2
LR          = 5e-3
WEIGHT_DECAY= 5e-4
EPOCHS      = 500
PATIENCE    = 50
CKPT = os.path.join(OUTPUT_DIR, "checkpoint.pt")
# ===================================

# 固定随机种子
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); dgl.seed(SEED)

def main():
    """
    训练主流程：
      1) 构图与数据划分 + 报表打印
      2) 初始化模型与优化器
      3) 训练循环（早停），记录验证曲线
      4) 载入最优模型，在测试集评估
      5) 保存并绘制曲线/曲面图
    """
    # 1) 构图 + 划分
    G, y, split, meta, n_inp = build_hetero_time_graph(NODES_CSV, EDGES_CSV)
    G = G.to(DEVICE)
    meta["y"] = y  # 给报表使用
    print_graph_report(meta)

    # 2) 模型（含对无特征节点的 nn.Embedding 注入）
    model = ProteinBinaryClassifier(G, n_inp=n_inp, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                                    time_window=TIME_WINDOW, dropout=DROPOUT, device=DEVICE, use_layernorm=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 划分索引与标签转设备
    train_idx = split["train_idx"].to(DEVICE)
    val_idx   = split["val_idx"].to(DEVICE)
    test_idx  = split["test_idx"].to(DEVICE)
    y = y.to(DEVICE)

    # 3) 训练
    print("开始训练 ...")
    # EarlyStopping：仅用验证损失触发，不额外打印中间 verbose
    es = EarlyStopping(patience=PATIENCE, verbose=False, path=CKPT, trace_func=lambda *args, **kwargs: None)

    # 记录容器（便于绘图）
    epochs = []
    val_auc, val_ap, val_f1, val_acc, val_bacc = [], [], [], [], []
    val_pp, val_rp, val_pn, val_rn = [], [], [], []

    for epoch in range(1, EPOCHS+1):
        # ---- 训练步 ----
        model.train()
        logit = model(G)  # (N_protein,)
        loss = F.binary_cross_entropy_with_logits(logit[train_idx], y[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- 验证步 ----
        model.eval()
        with torch.no_grad():
            v_logit = model(G)  # ← 不再重复 .to(DEVICE)
            v_prob = torch.sigmoid(v_logit)[val_idx].detach().cpu().numpy()
            v_true = y[val_idx].detach().cpu().numpy()
            # 验证损失（用于早停）
            val_loss = F.binary_cross_entropy_with_logits(v_logit[val_idx], y[val_idx])
            m = compute_all_metrics(v_true, v_prob)
            es(val_loss.item(), model)  # ← 用验证损失驱动早停

        # 每 10 个 epoch 打印一行简要指标
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | loss {loss.item():.4f} | AUC {m['AUC']:.4f} AP {m['AP']:.4f} F1(m) {m['F1m']:.4f} | "
                  f"ACC {m['ACC']:.4f} BACC {m['BACC']:.4f} | [+] P {m['P_pos']:.4f} R {m['R_pos']:.4f} | "
                  f"[-] P {m['P_neg']:.4f} R {m['R_neg']:.4f}")

        # 追加到曲线缓存
        epochs.append(epoch)
        val_auc.append(m["AUC"]); val_ap.append(m["AP"]); val_f1.append(m["F1m"])
        val_acc.append(m["ACC"]); val_bacc.append(m["BACC"])
        val_pp.append(m["P_pos"]); val_rp.append(m["R_pos"]); val_pn.append(m["P_neg"]); val_rn.append(m["R_neg"])

        if es.early_stop:
            print(f"触发早停：Epoch {epoch}（patience={PATIENCE}）")
            break

    # 4) 载入早停最优并在测试集评估
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    model.eval()
    with torch.no_grad():
        prob_test = torch.sigmoid(model(G.to(DEVICE)))[test_idx].detach().cpu().numpy()
        y_test    = y[test_idx].detach().cpu().numpy()
        mt = compute_all_metrics(y_test, prob_test)

    # 5) 可视化（验证集曲线 & 测试集 ROC/PR）
    plot_lines(epochs, {"AUC": val_auc, "AP": val_ap},
               os.path.join(OUTPUT_DIR, "val_auc_ap.png"), "AUC / AP")
    plot_lines(epochs, {"ACC": val_acc, "BACC": val_bacc},
               os.path.join(OUTPUT_DIR, "val_acc_bacc.png"), "Accuracy")
    plot_lines(epochs, {"F1(macro)": val_f1},
               os.path.join(OUTPUT_DIR, "val_f1_macro.png"), "F1 (macro)")
    plot_lines(epochs, {"P(+)": val_pp, "P(-)": val_pn},
               os.path.join(OUTPUT_DIR, "val_precision_by_class.png"), "Precision")
    plot_lines(epochs, {"R(+)": val_rp, "R(-)": val_rn},
               os.path.join(OUTPUT_DIR, "val_recall_by_class.png"), "Recall")
    plot_roc_pr(y_test, prob_test,
                os.path.join(OUTPUT_DIR, "test_roc_curve.png"),
                os.path.join(OUTPUT_DIR, "test_pr_curve.png"))

    # 6) 打印测试集表现
    print("==== 测试集表现（protein 节点） ====")
    print(f"AUC: {mt['AUC']:.4f}")
    print(f"AP : {mt['AP']:.4f}")
    print(f"ACC: {mt['ACC']:.4f}")
    print(f"BACC: {mt['BACC']:.4f}")
    print(f"F1 : {mt['F1m']:.4f}")
    print("已保存图像：")
    print("  val_auc_ap.png, val_acc_bacc.png, val_f1_macro.png, val_precision_by_class.png, val_recall_by_class.png")
    print("  test_roc_curve.png, test_pr_curve.png")

if __name__ == "__main__":
    main()