import random
from typing import List, Tuple, Dict

import numpy as np
import torch
from gensim.models import KeyedVectors
from torch.utils.data import Dataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_txt(path: str) -> Tuple[List[List[str]], List[int]]:
    texts, labels = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(" ", 1)
            label = int(parts[0])
            words = parts[1].split("\t") if len(parts) > 1 else []

            labels.append(label)
            texts.append(words)

    return texts, labels


def build_vocab(all_texts, min_freq=1) -> Dict[str, int]:
    freq = {}

    for sent in all_texts:
        for word in sent:
            freq[word] = freq.get(word, 0) + 1

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, count in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def build_embedding_matrix(vocab, w2v_path, embed_dim=50):
    print("Loading word2vec:", w2v_path)

    w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    embedding = np.random.normal(
        0,
        0.1,
        size=(len(vocab), embed_dim)
    ).astype(np.float32)

    embedding[vocab["<PAD>"]] = np.zeros(embed_dim, dtype=np.float32)

    hit = 0

    for word, idx in vocab.items():
        if word in w2v:
            embedding[idx] = w2v[word]
            hit += 1

    print(f"Embedding hit: {hit}/{len(vocab)}")

    return embedding


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.ids = [self.encode(words) for words in texts]

    def encode(self, words):
        ids = [
            self.vocab.get(word, self.vocab["<UNK>"])
            for word in words[:self.max_len]
        ]

        ids += [self.vocab["<PAD>"]] * (self.max_len - len(ids))

        return ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.ids[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )