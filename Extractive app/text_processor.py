import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextProcessor:
    def __init__(self, w2v_vocab, max_len):
        self.w2v_vocab = w2v_vocab
        self.max_len = max_len

    def tokenize_and_convert_to_sequences(self, sentences):
        sequences = []
        for sentence in sentences:
            tokenized = word_tokenize(sentence.lower())
            sequence = [self.w2v_vocab.get(word, 0) for word in tokenized]
            sequences.append(sequence)
        return sequences

    def pad_sequences(self, sequences):
        return pad_sequences(sequences, maxlen=self.max_len, padding='post')
