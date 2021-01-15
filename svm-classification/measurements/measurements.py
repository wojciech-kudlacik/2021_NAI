"""
weight-height SVM Classification
Authors: Wojciech Kudłacik, Norbert Daniluk
This program classifies data from the weight-height Dataset.
Gender is being determined by the provided height and weight.
Link: https://www.kaggle.com/mustafaali96/weight-height
It utilizes sklearn library as the primary engine to perform classification.
"""

from sklearn import svm
import pandas as pd


def get_measurements():
    """
    Asks a user for sample height and weight (in inches and pounds)
    """
    a = input("Height [inches]: ")
    b = input("Weight [pounds]: ")
    return [a, b]


class GenderClassification:
    """
    Main class to perform SVM Gender Classification.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Gets a data file and setups arrays from it
        """
        self.measurements_dataset = data
        self.measurements = self.measurements_dataset.iloc[:, 1:].values
        self.genders = self.measurements_dataset.iloc[:, 0].values
        self.clf = svm.SVC(gamma='scale')

    def run(self) -> None:
        """
        Entry point of the classification procedure.
        """
        self.train()
        inputs = get_measurements()
        result = self.make_prediction(inputs)
        self.print_result(result)

    def train(self) -> None:
        """
        Learn algorithm the provided data
        """
        self.clf = self.clf.fit(self.measurements, self.genders)

    def make_prediction(self, measurements) -> str:
        """
        Makes a prediction based on the previously learnt data and provided input from a user
        """
        return self.clf.predict([measurements])

    def print_result(self, result: str) -> None:
        """
        Prints a result.
        """
        print(result)


if __name__ == '__main__':
    data = pd.read_csv('measurements.csv')
    GenderClassification(data).run()
