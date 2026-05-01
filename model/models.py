import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        num_classes=2,
        filter_sizes=(2, 3, 4),
        num_filters=100,
        dropout=0.5
    ):
        super().__init__()

        vocab_size, embed_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix),
            freeze=False,
            padding_idx=0
        )

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim))
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)

        outputs = []

        for conv in self.convs:
            y = torch.relu(conv(x)).squeeze(3)
            y = torch.max_pool1d(y, y.size(2)).squeeze(2)
            outputs.append(y)

        x = torch.cat(outputs, dim=1)
        x = self.dropout(x)

        return self.fc(x)


class TextRNN(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        hidden_size=128,
        num_layers=1,
        bidirectional=True,
        dropout=0.5,
        num_classes=2
    ):
        super().__init__()

        vocab_size, embed_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix),
            freeze=False,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        output_dim = hidden_size * 2 if bidirectional else hidden_size

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        output, (hidden, cell) = self.lstm(x)

        if self.lstm.bidirectional:
            final = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final = hidden[-1]

        final = self.dropout(final)

        return self.fc(final)


class TextMLP(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        hidden_size=128,
        dropout=0.5,
        num_classes=2
    ):
        super().__init__()

        vocab_size, embed_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix),
            freeze=False,
            padding_idx=0
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        emb = self.embedding(x)

        mask = (x != 0).float().unsqueeze(-1)

        avg = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return self.classifier(avg)