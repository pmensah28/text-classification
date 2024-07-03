from typing import List, Tuple, Dict
from models import NaiveBayes

def compute_accuracy(model: NaiveBayes, valid_data: List[Tuple[str, List[str]]]) -> float:
    correct_predictions = 0
    for label, sentence in valid_data:
        predicted_label = model.predict(sentence)
        if predicted_label == label:
            correct_predictions += 1

    accuracy = correct_predictions / len(valid_data)
    return accuracy
