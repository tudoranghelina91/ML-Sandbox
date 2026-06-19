from collections import Counter

import pandas as pd
import json
import spacy
from spacy_langdetect import LanguageDetector
from spacy.language import Language

class VocabularyBuilder:
    language_codes = ["en"]

    @Language.factory("language_detector")
    def create_lang_detector(nlp, name):
        return LanguageDetector()
    
    def __init__(self, dictionary_path=None, df=None, save_file=False):
        if not dictionary_path and df is None:
            raise RuntimeError("Specify dictionary path, or csv")

        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("language_detector", last=True)
        
        if df is not None:
            self.__vocab = {}
            self.__build_vocab(df=df)
            if save_file:
                self.__save(dictionary_path=(dictionary_path if dictionary_path else "./data/vocab.json"))

        if dictionary_path:
            with open(dictionary_path, 'r') as fp:
                self.__vocab = json.load(fp)
            return

    def __build_vocab(self, df, min_freq=1):
        counter = Counter()

        for text in df["Comment"]:
            if not isinstance(text, str):
                text = ""

            doc = self.nlp(text)
            if not self.__is_english(doc):
                continue

            tokens = [t.text.lower() for t in doc]
            counter.update(tokens)

        self.__vocab = {"<pad>": 0, "<unk>": 1}
        for token, freq in counter.items():
            if freq >= min_freq:
                self.__vocab[token] = len(self.__vocab)
    
    def __save(self, dictionary_path):
        with open(dictionary_path, 'w') as fp:
            json.dump(self.__vocab, fp, indent=4)

    def __is_english(self, doc):
        return doc._.language["language"] in self.language_codes and doc._.language["score"] > 0.7

    def build(self):
        return self.__vocab