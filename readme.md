# Sentiment Analysis (Text Classification)

一个基于 PyTorch 的情感分析项目，实现 CNN / BiLSTM / MLP / Attention 模型，并结合大模型生成数据进行鲁棒性分析。

---

## Project Structure

```
.
├── Dataset/                 # 原始数据 + LLM生成数据
├── Dataset_llm/             # LLM划分后的训练/验证集
├── model/
│   ├── config.py           # 参数配置
│   ├── data_trans.py       # 数据处理
│   ├── models.py           # CNN / RNN / MLP / Attention
│   ├── train.py            # 训练与评估
│   ├── main.py             # 主程序入口
│   ├── plot_results.py     # 结果可视化
│   └── generate_robust_data_deepseek.py  # LLM数据生成
├── results/                 # 实验日志（自动生成）
└── README.md
```

---

## Features

* 多模型实现：CNN / BiLSTM / MLP / Attention
* 支持 Word2Vec 预训练 embedding
* 自动日志记录（每次训练独立目录）
* 参数调优对比（lr / dropout）
* 可视化脚本（自动画图）
* 大模型生成数据（DeepSeek API）
* 鲁棒性分析（分布迁移）

---

## Environment

```bash
conda create -n sentiment python=3.10
conda activate sentiment
pip install torch numpy pandas matplotlib scikit-learn tqdm
```
```bash
pip install numpy==1.26.4
```

---

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python model/main.py \
  --data_dir ./Dataset \
  --epochs 8 \
  --batch_size 32 \
  --lr 0.0005 \
  --dropout 0.5
```

---

## Generate LLM Data

```bash
export DEEPSEEK_API_KEY=your_key
python model/generate_robust_data_deepseek.py
```

---

## Visualization

```bash
python model/plot_results.py
```

生成：

```
results/figures/
```

---

## Results Summary

| Model     | F1 (Original) | F1 (LLM Data) |
| --------- | ------------- | ------------- |
| CNN       | ~0.85         | ~0.86         |
| BiLSTM    | ~0.80         | **~0.92**     |
| MLP       | ~0.85         | ~0.89         |
| Attention | ~0.84         | ~0.68         |

---

## Key Insights

* CNN / MLP 在真实数据上更稳定
* BiLSTM 在结构化（LLM）数据上更强
* Attention 对数据分布敏感
* 模型性能强依赖数据分布

---

## Robustness Evaluation

使用大模型生成数据：

* 否定句（不是很好）
* 转折句（虽然...但是...）
* 口语表达
