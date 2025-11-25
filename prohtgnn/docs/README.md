# HTGNN-Protein-Binary-Classification (异构时态图 · 蛋白质重要性预测)

本项目基于 **HTGNN**（Heterogeneous Temporal Graph Neural Network）在自建异构时态图（`nodes.csv` + `edges.csv`）上完成 **protein 节点二分类**（重要蛋白质预测）。实现 **不改动 HTGNN 主干机制**，仅在外层封装：数据读入/构图、无特征节点的可训练嵌入、二分类头与指标/可视化；超参与训练流程与论文/官方实现保持一致（Adam, lr=5e‑3, wd=5e‑4, dropout=0.2, 2 层, 500 epoch, 早停耐心 50）。

HTGNN 的单层包含三段聚合：**Intra‑relation（同关系）→ Inter‑relation（跨关系）→ Across‑time（跨时间）**，并带**门控残差**；多层堆叠后将所有时间片表示**求和**得到最终节点表征。项目为无特征类型（`GO`、`reaction`）注入 `nn.Embedding` 并统一经 HTGNN 主干，最终在 **protein** 上接两层 MLP 输出 logits（未过 Sigmoid）。

---

## 目录结构

```
.
├── data/
│   ├── nodes.csv
│   └── edges.csv
├── outputs/                         # 训练产物（曲线图与 checkpoint）
├── preprocess.py                    # 读 CSV，解析 timeset -> 36 个时间片
├── dataset.py                       # 组装 DGL 异构时态图 + 分层划分
├── model.py                         # 复用 HTGNN 主干 + 无特征类型嵌入 + 二分类头
├── train.py                         # 训练/验证/早停/测试与可视化（集中式配置）
├── metrics.py                       # AUC/AP/ACC/BACC/F1 与分类别 P/R + 绘图
└── utils_report.py                  # 构图报表（严格对齐输出格式）
```

- 训练配置集中在 `train.py` 顶部（不使用命令行参数），默认 `DEVICE = cuda:5`、`TIME_WINDOW = 36`、`D_MODEL = 32` 等，可改为本机可用设备与路径。固定随机种子 42。
- 早停沿用原仓库 `EarlyStopping`，以**验证损失**为监控，保存到 `outputs/checkpoint.pt`。

---

## 运行环境

- Python ≥ 3.8  
- PyTorch 1.8.1、DGL 0.6.0、NumPy、Pandas、scikit‑learn、matplotlib

---

## 数据格式

### `nodes.csv`（必需列）
- `Id`（字符串）：节点唯一标识  
- `function`：对 **protein**，`ORF-V` 记为 1，否则 0；其他类型可留空  
- `t0`~`t35`：36 列时序表达（仅 **protein**/**single protein** 有值；`GO`/`reaction` 为空）  
- `node type`：节点类型（`protein` / `GO` / `reaction` / `single protein`；其中 `single protein` 归并到 `protein`）  

### `edges.csv`（必需列）
- `Source` / `Target`（字符串）：端点 Id  
- `Type`：`Directed` / `Undirected`（否则报错）  
- `timeset`：半开区间串（例：`[a,b]; [c,d]; ...`）；**唯一端点=37** ⇒ 36 个 `[t_i,t_{i+1})` 时间片；空值表示**全时段存在**  
- `edge_type`：基础关系名（如 `protein-protein`、`enables` 等）  

> **构图策略**：将每条边展开到覆盖的时间片，并把基础关系名**加上 `_t{idx}` 后缀**作为 canonical etype；**Undirected** 边补反向一条。

---

## 一键运行

1. 将 `nodes.csv`、`edges.csv` 放入 `./data/`  
2. 打开 `train.py` 顶部，按需修改 `DEVICE`/路径/隐维等（默认配置与论文/官方一致）  
3. 运行：
   ```bash
   python train.py
   ```

流程：加载数据并构图 → 打印**Graph Report**（严格对齐格式）→ 训练（每 10 epoch 打印验证指标，仅 protein）→ 早停与最优模型保存 → 在**测试集（protein）**评估 + 绘制 5 张验证折线与 2 张测试曲线图（保存到 `outputs/`）。

---

## 输出示例（严格照抄）

```
构建异构时态图 ...
========== Graph Report ==========
[Nodes] total = 6,086
  - GO        :     400 (  6.57%)
  - protein   :   5,003 ( 82.21%)
  - reaction  :     683 ( 11.22%)
[Labels on protein] total = 5,003
  - positive :   4,668 ( 93.30%)
  - negative :     335 (  6.70%)

[Edges] total = 57,854
  Aggregated by base relation:
    * (protein, protein-protein          , protein ) : 36,873
    * (protein, co_localized             , protein ) : 16,356
    * (protein, protein-reaction         , reaction) : 1,886
    * (protein, co_reaction              , protein ) : 1,044
    * (protein, enables                  , GO      ) : 663
    * (protein, located_in               , GO      ) : 567
    * (protein, involved_in              , GO      ) : 361
    * (protein, acts_upstream_of_or_within, GO      ) : 104
[Splits] size and ratio over labeled nodes
  - train: 3,502 ( 70.00%) | [+] 3,268 ( 93.32%)  [-]   234 (  6.68%)
  - val  :   750 ( 14.99%) | [+]   700 ( 93.33%)  [-]    50 (  6.67%)
  - test :   751 ( 15.01%) | [+]   700 ( 93.21%)  [-]    51 (  6.79%)
==================================
开始训练 ...
Epoch 001 | loss 0.4622 | AUC 0.5000 AP 0.9333 F1(m) 0.4828 | ACC 0.9333 BACC 0.5000 | [+] P 0.9333 R 1.0000 | [-] P 0.0000 R 0.0000
Epoch 010 | loss 0.8985 | AUC 0.7958 AP 0.9688 F1(m) 0.4828 | ACC 0.9333 BACC 0.5000 | [+] P 0.9333 R 1.0000 | [-] P 0.0000 R 0.0000
Epoch 020 | loss 0.2815 | AUC 0.8006 AP 0.9723 F1(m) 0.8538 | ACC 0.9667 BACC 0.8243 | [+] P 0.9760 R 0.9886 | [-] P 0.8049 R 0.6600
==== 测试集表现（protein 节点） ====
AUC: 0.7810
AP : 0.9707
ACC: 0.9587
BACC: 0.8233
F1 : 0.8324
已保存图像：
  val_auc_ap.png, val_acc_bacc.png, val_f1_macro.png, val_precision_by_class.png, val_recall_by_class.png
  test_roc_curve.png, test_pr_curve.png
```

---

## 评估指标与曲线

仅在 **protein** 节点上计算：**AUC、AP、ACC、BACC、F1(macro)**，并分别给出**正/负类 Precision/Recall**；每 10 epoch 打印；保存验证折线 5 张与测试 **ROC/PR** 2 张图。AUC/AP 对“单类 split”做 NaN 兼容；分类阈值为 0.5（可按需外部搜索最优阈值）。

---

## 许可证与引用

- 方法与配置参考 HTGNN 论文（框架与三类聚合见 Fig.2；训练细节 §5.3）。
- 构图「时间后缀化关系名」策略与原仓库数据管线一致（COVID/MAG）。
