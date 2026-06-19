import pandas as pd
import spacy
import torch
import multiprocessing

from collections import Counter
from torch.utils.data import DataLoader
from SimpleSentimentNeuralNetwork import SimpleSentimentNeuralNetwork
from VocabularyBuilder import VocabularyBuilder
from YouTubeCommentsDataset import YouTubeCommentsDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from multiprocessing import Pool

def process_chunk(df):
    # clean Comment column
    df["Comment"] = df["Comment"].fillna("").astype(str)

    # map text labels → integers
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["Sentiment"] = (
        df["Sentiment"]
        .fillna("neutral")          # fallback
        .str.lower()                # normalize
        .map(label_map)             # convert to int
        .astype(int)
    )
    return df

def load_csv(csv_path):
    chunks = pd.read_csv(csv_path, chunksize=2000)
    with Pool() as pool:
        results = pool.map(process_chunk, chunks)

    df = pd.concat(results)
    return df

def train_per_epoch(epoch):
    print(f"Training: Epoch {epoch + 1}")
    losses = []
    all_preds = []
    all_labels = []
    model.train()
    dataset.train()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        all_preds.extend(logits.argmax(dim=1).tolist())
        all_labels.extend(y.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"Training: Epoch {epoch+1} | Loss={sum(losses)/len(losses):.4f} | Acc={acc:.3f} | F1={f1:.3f}")

def validate_per_epoch(epoch):
    print(f"Validating: Epoch {epoch + 1}")
    dataset.validate()
    model.eval()
    losses = []
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y.cpu().numpy())

    acc = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average="weighted")
    macro_f1 = f1_score(val_labels, val_preds, average="macro")
    cm = confusion_matrix(val_labels, val_preds, labels=[0, 1, 2])

    print(f"Validation: Loss={sum(losses)/len(losses):.4f} | Acc={acc:.3f} | Weighted F1={f1:.3f} | Macro F1={macro_f1:.3f}")
    print("Confusion matrix (rows = true, cols = pred):")
    print(cm)

    print("\nClassification report:")
    print(classification_report(val_labels, val_preds, labels=[0, 1, 2],
                                target_names=["neg", "neu", "pos"]))

def train_model():
    print("Training...")
    for epoch in range(8):
        train_per_epoch(epoch)
        validate_per_epoch(epoch)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    print(f"Device used: {device.type}")

    csv_path = "./data/YoutubeCommentsDataSet.csv"
    df = load_csv("./data/YoutubeCommentsDataSet.csv")
    
    vocab_builder = VocabularyBuilder(df=df, save_file=True)
    vocab = vocab_builder.build()
    dataset = YouTubeCommentsDataset(df, vocab)

    # dataset + loader
    dataset = YouTubeCommentsDataset(df, vocab, device=device)
    dataset.train()
    loader = DataLoader(dataset, batch_size=128 if device.type == "cuda" else 32, shuffle=True, generator=torch.Generator("cuda" if torch.cuda.is_available() else "cpu"))

    # Compute weights using the actual number of data classification by invers freq weighting.
    # This will punsish the model
    # Your real class counts - we use this for weights
    counts = torch.tensor([475, 937, 2270], dtype=torch.float32)
    N = counts.sum()
    K = len(counts)

    # Inverse frequency weighting
    weights = N / (K * counts)
    weights = weights.to(device)

    # model
    model = SimpleSentimentNeuralNetwork(vocab_size=len(vocab))
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)

    # training
    train_model()

    # save model
    torch.save(model.state_dict(), "sentiment_model.pt")

