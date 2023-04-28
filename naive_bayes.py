import csv
import numpy as np
import random

class NaiveBayesClassifier:
    def __init__(self, filename, shuffle=True):
        self.dataset = []
        self.shuffle = shuffle
        with open(filename) as csvfile:
            for row in csv.reader(csvfile):
                self.dataset.append(row)

    def train_and_test(self, test_train_ratio):
        self.clean_dataset()
        train_dataset = self.dataset[:round(test_train_ratio * len(self.dataset))]
        test_dataset = self.dataset[round(test_train_ratio * len(self.dataset)):]
        total_positives = sum([x[-1] == '1' for x in train_dataset])
        total_negatives = sum([x[-1] == '0' for x in train_dataset])
        accuracy = 0
        for row in test_dataset:
            positive_score = total_positives / len(train_dataset)
            negative_score = total_negatives / len(train_dataset)
            for i in range(len(self.dataset[0]) - 1):
                positive_score *= sum([x[i] == row[i] and x[-1] == '1' for x in train_dataset]) / total_positives
                negative_score *= sum([x[i] == row[i] and x[-1] == '0' for x in train_dataset]) / total_negatives
            if positive_score > negative_score and row[-1] == '1':
                accuracy += 1
            elif positive_score < negative_score and row[-1] == '0':
                accuracy += 1

        test_positives = sum([x[-1] == '1' for x in test_dataset])
        test_negatives = sum([x[-1] == '0' for x in test_dataset])
        print(f"Total Correct: {accuracy}")
        print(f"Size of test dataset: {len(test_dataset)}")
        print(f"Accuracy: {accuracy / len(test_dataset)}")
        print(f"Number of 1 labels: {test_positives}")
        print(f"Number of 0 labels: {test_negatives}")

    def clean_dataset(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        for i in range(len(self.dataset[0]) - 1):
            # If there are less than 12 unique values in column, considered categorical
            if len(np.unique([x[i] for x in self.dataset])) < 12:
                continue
            # Convert continuous variables to categorical
            bottom = min([float(x[i]) for x in self.dataset])
            top = max([float(x[i]) for x in self.dataset])
            middle = (top + bottom) / 2
            bm = (bottom + middle) / 2
            tm = (top + middle) / 2    
            for row in self.dataset:
                if float(row[i]) <= bm:
                    row[i] = '1'
                elif float(row[i]) <= middle:
                    row[i] = '2'
                elif float(row[i]) <= tm:
                    row[i] = '3'
                else:
                    row[i] = '4'
        
        # Map labels to binary '0' and '1'
        labels = np.unique([x[-1] for x in self.dataset])
        for row in self.dataset:
            if row[-1] == labels[0]:
                row[-1] = '0'
            elif row[-1] == labels[1]:
                row[-1] = '1'
            else:
                print("LABELS NOT BINARY!")
                exit()

def main():
    nb = NaiveBayesClassifier('Datasets/diabetes.csv')
    nb.train_and_test(0.67)
    print()
    nb = NaiveBayesClassifier('Datasets/indian-diabetes.csv')
    nb.train_and_test(0.67)
    print()
    nb = NaiveBayesClassifier('Datasets/banknote.csv')
    nb.train_and_test(0.67)
    print()
    nb = NaiveBayesClassifier('Datasets/breast-cancer.csv', shuffle=False)
    nb.train_and_test(0.67)

if __name__ == "__main__":
    main()
