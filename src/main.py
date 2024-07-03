from data_loader import read_data
from models import NaiveBayes
from evaluate import compute_accuracy

def main():
    # Load data
    train_data = read_data('../data/train1.txt')
    valid_data = read_data('../data/valid1.txt')

    # Initialize and train the Naive Bayes model
    nb_model = NaiveBayes(mu=1.0)
    nb_model.fit(train_data)

    # Compute accuracy on validation set
    accuracy = compute_accuracy(nb_model, valid_data)
    print(f"Accuracy on validation set: {accuracy:.2%}")

if __name__ == "__main__":
    main()
