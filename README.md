# SGL-ReChorus

本项目为中山大学机器学习课程作业，基于 ReChorus 框架复现了 SGL 模型。SGL 通过引入自监督学习（对比学习）辅助任务，增强了图神经网络在推荐系统中的节点表示能力。


## SGL推荐算法

SGL 在标准的 LightGCN 骨架上增加了一个辅助的自监督学习任务。通过对用户-物品二部图进行数据增强（如随机丢弃边或节点），模型生成不同的视图（Views），并利用对比学习（InfoNCE Loss）最大化同一节点在不同视图下表示的一致性。

### 模型特性
*   **骨干网络**: LightGCN (轻量级图卷积网络)。
*   **多任务学习**: 总损失 = BPR 推荐损失 + `ssl_reg` * SSL 对比损失。
*   **框架适配**: 
    *   完全适配 ReChorus 的 `BaseRunner` 训练流程。
    *   **重要**: L2 正则化由优化器（Adam）的 `weight_decay` 自动处理，**避免了在 Loss 函数中重复计算导致的双重正则化问题**。
*   **数据流**: `forward` 函数同时计算预测分数和 SSL Loss，并通过 `out_dict` 传递给 `loss` 函数进行汇总。

## ReChorus框架
ReChorus2.0 是一个模块化、任务灵活的 PyTorch 推荐库，专为科研用途而设计。它旨在为研究人员提供一个灵活的框架，用于实现各种推荐任务、比较不同的算法，并适应多样化和高度定制化的数据输入。ReChorus框架分为Input、Reader、Model和Runner层。其中，本次作业在`src/models/general`中增加了`SGL.py`，利用`forward()`传递数据，契合原ReChorus结构。
+ [Rechorus](https://github.com/THUwangcy/ReChorus)


## 运行指引
本实验在`python3.8`环境下进行。
1. 安装[Anaconda](https://docs.conda.io/en/latest/miniconda.html) with Python = 3.8.20
2. 克隆储存库
   ```bash
    git clone https://github.com/Zhangmia-ops/SGL-on-Rechorus
    ```
4. 安装所需组件
   ```bash
    cd ReChorus
    pip install -r requirements.txt
    ```
5. 使用内置数据集运行模型
   
   ```bash
    python src/main.py --model_name SGL --dataset MovieLens_1M/ml-1m --lr 1e-2  --epoch 30 --num_workers 0 --test_all 1 --emb_size 64 --l2 1e-5 --batch_size 2048 --n_layers 2
   ```
   可以更改参数复现报告中各项实验的SGL表现，


