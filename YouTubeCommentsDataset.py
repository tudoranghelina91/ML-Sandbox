from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import pandas as pd
import torch

from TextEncoder import TextEncoder

class YouTubeCommentsDataset(Dataset):
    def __init__(self, df, vocab, max_len=128, device: torch.device = torch.device("cpu"), is_training=True):
        self.device = device
        self.encoder = TextEncoder(vocab, max_len, device)
        self.is_training = is_training
        
        self.train_df, self.val_df = train_test_split(df, test_size=0.2, random_state=42)

    def __len__(self):
        if self.is_training:
            return len(self.train_df)
        return len(self.val_df)
    
    def train(self):
        self.is_training = True

    def validate(self):
        self.is_training = False

    def __getitem__(self, idx):
        row = None
        if self.is_training:
            row = self.train_df.iloc[idx]
        else:
            row = self.val_df.iloc[idx]
        x = self.encoder.encode(row["Comment"])
        y = torch.tensor(row["Sentiment"], dtype=torch.long, device=self.device)
        return x, y