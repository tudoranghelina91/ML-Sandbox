import torch
from SimpleSentimentNeuralNetwork import SimpleSentimentNeuralNetwork
from VocabularyBuilder import VocabularyBuilder
from TextEncoder import TextEncoder

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

vocab = VocabularyBuilder(dictionary_path="./data/vocab.json").build()

label_map = {0: "negative", 1: "neutral", 2: "positive"}

model = SimpleSentimentNeuralNetwork(vocab_size=len(vocab))
model.load_state_dict(torch.load("./sentiment_model.pt"))
model.eval()

with torch.no_grad():
    encoder = TextEncoder(vocab)
    for word in [
        "good",
        "bad",
        "okay",
        "meh",
        "this is fine",
        "terrible",
        "this is the best thing in the world",
        "why would you say this movie is great? it obviously isn't! it's simply awful!",
        "shit",
        "o",
        "I am enabled to add value.",
        "You, sir, are the best!",
        "You are dead to me!",
        "I hate your guts",
        "Someone is trying to murder me"
    ]:
        text_tensors = encoder.encode(word).unsqueeze(0)
        output = model(text_tensors)
        result = torch.argmax(output, dim=1)
        print(f"{word}: {label_map[int(result)]}")