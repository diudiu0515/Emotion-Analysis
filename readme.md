# Sentiment Analysis (Text Classification)

一个基于 PyTorch 的情感分析项目，实现了多种模型（CNN / BiLSTM / MLP / Attention），并支持实验日志记录与结果对比分析。

## Project Structure

```
.
├── Dataset/              # 数据集
├── model/
│   ├── config.py        # 参数配置
│   ├── data_trans.py    # 数据处理（读取、词表、embedding）
│   ├── models.py        # 模型定义（CNN / RNN / MLP / Attention）
│   ├── train.py         # 训练与评估逻辑
│   ├── logger.py        # 实验日志记录
│   └── main.py          # 主程序（调度训练流程）
├── results/             # 实验结果（按时间自动生成）
└── README.md
```


## Models

本项目实现了四种文本分类模型：

* **TextCNN**：卷积网络，捕捉 n-gram 局部特征
* **BiLSTM**：双向 LSTM，建模上下文依赖
* **MLP**：平均词向量的 baseline 模型
* **Attention**：基于注意力机制的加权表示


## Method Overview

* 使用 **Word2Vec** 预训练词向量
* 文本 → embedding → 模型 → 分类
* 使用以下指标评估模型性能：

```
Accuracy / Precision / Recall / F1-score
```

## Environment

推荐使用 conda：

```bash
conda create -n sentiment python=3.10
conda activate sentiment
pip install torch numpy scikit-learn tqdm
```
注意：

```bash
pip install numpy==1.26.4
```
避免 numpy 2.x 与 PyTorch 不兼容问题。


## Usage

运行训练：

```bash
CUDA_VISIBLE_DEVICES=0 python model/main.py --epochs 5
```

常用参数：

```
--epochs        训练轮数
--batch_size    batch 大小
--lr            学习率
--dropout       dropout 比例
--max_len       最大句长
--min_freq      词频阈值
```

---

## Experiment Logging

每次运行会自动生成一个时间戳文件夹：

```
results/YYYY-MM-DD_HH-MM/
├── config.json    # 本次实验参数
└── metrics.csv    # 各模型测试结果
```

示例：

```
model,accuracy,precision,recall,f1,loss
CNN,0.8455,...
BiLSTM,0.8130,...
MLP,0.8537,...
Attention,0.8455,...
```
