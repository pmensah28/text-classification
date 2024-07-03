import math
from typing import List, Tuple, Dict
from collections import defaultdict

class NaiveBayes:
    def __init__(self, mu: float = 1.0):
        self.mu = mu
        self.n_examples = 0
        self.n_words_per_label = defaultdict(lambda: 0)
        self.label_counts = defaultdict(lambda: 0)
        self.word_counts = defaultdict(lambda: defaultdict(lambda: 0.0))

    def fit(self, data: List[Tuple[str, List[str]]]):
        for label, sentence in data:
            self.n_examples += 1
            self.label_counts[label] += 1
            self.n_words_per_label[label] += len(sentence)
            for word in sentence:
                self.word_counts[label][word] += 1

    def predict(self, sentence: List[str]) -> str:
        best_label = None
        best_score = float('-inf')

        for label in self.word_counts.keys():
            score = 0
            prior = self.label_counts[label] / sum(self.label_counts.values())
            score += math.log(prior)

            for word in sentence:
                word_count = self.word_counts[label][word]
                total_words = self.n_words_per_label[label]
                vocab_size = len(self.word_counts[label])

                word_probability = (self.mu + word_count) / (self.mu * vocab_size + total_words)
                score += math.log(word_probability)

            if score > best_score:
                best_score = score
                best_label = label

        return best_label
