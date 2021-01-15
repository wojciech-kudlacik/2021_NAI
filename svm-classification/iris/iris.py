"""
Iris Petal SVM Classification
Authors: Wojciech Kudłacik, Norbert Daniluk
This program classifies data from the Iris Dataset.
Type of an Iris flower is being determined by the length and width of its petals and sepals.
Link: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
It utilizes sklearn library as the primary engine to perform classification.
"""

from iris_setup import IrisSetup
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from typing import List


class IrisClassification:
    """
    Main class to perform SVM Iris Classification.
    """
    def __init__(self, data: pd.DataFrame) -> None:
        self.iris_dataset = data
        self.iris_measurements = self.iris_dataset.iloc[:, :-1]
        self.iris_target = self.iris_dataset.iloc[:, -1].values
        self.sepal_data = self.iris_dataset.iloc[:, :2]
        self.petal_data = self.iris_dataset.iloc[:, 2:]

    @staticmethod
    def get_user_input() -> List[str]:
        """
        Method to get input from User
        :return: List of user inputs, that describe Iris flower
        """
        sepal_len = input("Sepal Len [cm]: ")
        sepal_width = input("Sepal Width [cm]: ")
        petal_len = input("Petal Len [cm]: ")
        petal_width = input("Sepal Width [cm]: ")
        return [sepal_len, sepal_width, petal_len, petal_width]

    def visualize_data(self, iris_part: str) -> None:
        """
        Method to present correlation data between sepals or petals on a graph
        :param iris_part: Name of the iris part
        """
        if iris_part is "Sepal":
            x = self.sepal_data
        else:
            x = self.petal_data

        plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=self.replace_class_values(), cmap='coolwarm')
        plt.xlabel(f'{iris_part} len')
        plt.ylabel(f'{iris_part} width')
        plt.title(f'{iris_part} width & len correlation to Iris Type')
        plt.show()

    def replace_class_values(self) -> SVC:
        """
        Method to replace Iris Setosa names into 0, 1, 2 values for visualization purposes
        :return: Values of the last column of dataset as numbers
        """
        target_as_numbers = self.iris_dataset.replace({"class": {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}})
        return target_as_numbers.iloc[:, -1].values

    def visualize_heatmap(self) -> None:
        """
        Method to present the correlation of sepals or petals as a heatmap
        Utilizes seaborn package
        """
        sns.heatmap(self.iris_dataset.corr())
        plt.title('Correlation on iris classes')
        plt.show()

    def train(self) -> SVC:
        """
        Method to train the classifier based on the provided dataset
        :return: classifier: SVC model
        """
        x_train, x_test, y_train, y_test = train_test_split(self.iris_measurements, self.iris_target, test_size=0.25)
        classifier = SVC(kernel='linear')
        classifier.fit(x_train, y_train)
        return classifier

    @staticmethod
    def make_prediction(classifier, measurements) -> str:
        """
        Makes a prediction based on the previously learnt data and provided input from a user
        :param classifier: SVC model
        :param measurements: User input that describes Iris flower
        :return: Predicted Iris Flower class
        """
        return classifier.predict([measurements])

    def run(self) -> None:
        """
        Method to run the program
        """
        self.visualize_data("Sepal")
        self.visualize_data("Petal")
        self.visualize_heatmap()
        inputs = self.get_user_input()
        classifier = self.train()
        result = self.make_prediction(classifier, inputs)
        print(result)


if __name__ == '__main__':
    dataset = IrisSetup().create_dataset_from_file()
    IrisClassification(dataset).run()
