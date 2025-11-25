# -*- coding: utf-8 -*-
# 评估指标与可视化
# 说明：仅添加注释与文档字符串，不改动任何业务逻辑

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score
import matplotlib.pyplot as plt

def compute_all_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5):
    """
    仅在 protein 节点上计算常用二分类指标：
      - AUC, AP
      - ACC, BACC
      - F1(macro)
      - 正/负类的 Precision / Recall

    兼容性：
      - 若 split 中出现单类，AUC/AP 捕获异常返回 NaN
    """
    y_true = y_true.astype(int)
    y_prob = y_score.astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # 若某 split 出现单类（极端不平衡），AUC/AP 做兼容
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    try:
        ap = average_precision_score(y_true, y_prob)
    except Exception:
        ap = float("nan")

    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average="macro")
    bacc = balanced_accuracy_score(y_true, y_pred)

    # 分别计算正负类的精确率/召回率（负类通过 1-y 的方式计算）
    prec_pos = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_pos  = recall_score(y_true, y_pred,    pos_label=1, zero_division=0)
    prec_neg = precision_score(1-y_true, 1-y_pred, pos_label=1, zero_division=0)
    rec_neg  = recall_score(1-y_true,    1-y_pred, pos_label=1, zero_division=0)

    return {
        "AUC": auc, "AP": ap, "ACC": acc, "BACC": bacc, "F1m": f1m,
        "P_pos": prec_pos, "R_pos": rec_pos, "P_neg": prec_neg, "R_neg": rec_neg
    }

def plot_lines(xs, ys_dict, out_png, ylabel):
    """
    折线图绘制（训练过程指标）：
      xs: 横轴（epoch）
      ys_dict: {曲线名: y 序列}
      out_png: 输出路径
      ylabel : Y 轴标题
    """
    plt.figure(figsize=(6,4))
    for k, v in ys_dict.items():
        plt.plot(xs, v, label=k)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} per Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_roc_pr(y_true, y_prob, roc_png, pr_png):
    """
    绘制测试集 ROC / PR 曲线并保存为图片。
    说明：这里以可视化为主，AP 在图例中给出近似值（数值以 trapz 近似）。
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f'ROC Curve (AUC = {auc(fpr, tpr):.4f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_png, dpi=180); plt.close()
    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = np.trapz(prec[::-1], rec[::-1])  # 可视化展示
    plt.figure(figsize=(6,4))
    plt.plot(rec, prec, label=f"AP~{ap:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f'PR Curve (AP = {ap:.4f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_png, dpi=180); plt.close()
