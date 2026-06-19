import spacy
import torch

class TextEncoder:
    def __init__(self, vocab, max_len=128, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.nlp = spacy.load("en_core_web_sm")
        self.vocab = vocab
        self.max_len = max_len
        self.pad_id = vocab["<pad>"]
        self.unk_id = vocab["<unk>"]
        self.device=device

    def encode(self, text):
        doc = self.nlp(text)
        tokens = [t.text.lower() for t in doc]
        ids = [self.vocab.get(tok, self.unk_id) for tok in tokens][:self.max_len]
        ids += [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long, device=self.device)