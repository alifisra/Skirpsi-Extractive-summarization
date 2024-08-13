import os
from gensim.models import Word2Vec
import tensorflow as tf
import tf_keras

class ModelLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.skipGram_path = os.path.join(base_path, "skipgram.model")
        self.model_path = os.path.join(base_path, "M1.h5")
        self.load_models()

    def load_models(self):
        self.w2v_model = Word2Vec.load(self.skipGram_path)
        self.M1 = tf_keras.models.load_model(self.model_path)

    def get_w2v_vocab(self, vocab_size=250):
        return {word: idx + 1 for idx, word in enumerate(self.w2v_model.wv.index_to_key[:vocab_size - 1])}
