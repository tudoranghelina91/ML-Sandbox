import torch
import torch.nn as nn

class SimpleSentimentNeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        out, (h, c) = self.encoder(emb)

        h_forward = h[-2]
        h_backward = h[-1]

        pooled = torch.cat((h_forward, h_backward), dim=1)
        logits = self.classifier(pooled)
        return logits