import os

from torch.utils.data import DataLoader

from config import get_args
from data_trans import (
    set_seed,
    read_txt,
    build_vocab,
    build_embedding_matrix,
    SentimentDataset
)
from models import TextCNN, TextRNN, TextMLP
from train import train_model


def main():
    args = get_args()

    set_seed(args.seed)

    train_path = os.path.join(args.data_dir, "train.txt")
    val_path = os.path.join(args.data_dir, "validation.txt")
    test_path = os.path.join(args.data_dir, "test.txt")
    w2v_path = os.path.join(args.data_dir, "wiki_word2vec_50.bin")

    train_texts, train_labels = read_txt(train_path)
    val_texts, val_labels = read_txt(val_path)
    test_texts, test_labels = read_txt(test_path)

    vocab = build_vocab(
        train_texts + val_texts + test_texts,
        min_freq=args.min_freq
    )

    embedding_matrix = build_embedding_matrix(
        vocab,
        w2v_path,
        embed_dim=50
    )

    train_dataset = SentimentDataset(
        train_texts,
        train_labels,
        vocab,
        args.max_len
    )

    val_dataset = SentimentDataset(
        val_texts,
        val_labels,
        vocab,
        args.max_len
    )

    test_dataset = SentimentDataset(
        test_texts,
        test_labels,
        vocab,
        args.max_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    results = {}

    cnn = TextCNN(
        embedding_matrix,
        filter_sizes=(2, 3, 4),
        num_filters=100,
        dropout=args.dropout
    )

    results["CNN"] = train_model(
        cnn,
        train_loader,
        val_loader,
        test_loader,
        args,
        "CNN"
    )

    rnn = TextRNN(
        embedding_matrix,
        hidden_size=128,
        bidirectional=True,
        dropout=args.dropout
    )

    results["BiLSTM"] = train_model(
        rnn,
        train_loader,
        val_loader,
        test_loader,
        args,
        "BiLSTM"
    )

    mlp = TextMLP(
        embedding_matrix,
        hidden_size=128,
        dropout=args.dropout
    )

    results["MLP"] = train_model(
        mlp,
        train_loader,
        val_loader,
        test_loader,
        args,
        "MLP"
    )

    print("\n===== Final Results =====")
    print("Model\tAccuracy\tPrecision\tRecall\tF1")

    for name, m in results.items():
        print(
            f"{name}\t"
            f"{m.acc:.4f}\t"
            f"{m.precision:.4f}\t"
            f"{m.recall:.4f}\t"
            f"{m.f1:.4f}"
        )


if __name__ == "__main__":
    main()