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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


class IrisClassification:
    """
    Main class to perform SVM Iris Classification.
    """
    def __init__(self, data: pd.DataFrame) -> None:
        self.iris_dataset = data
        self.iris_target = self.iris_dataset.iloc[:, -1].values

    def visualize_data(self, iris_part: str) -> None:
        """
        Method to present correlation data between sepals or petals on a graph
        :param iris_part: Name of the iris part
        """
        if iris_part is "Sepal":
            x = self.iris_dataset.iloc[:, :2]
        else:
            x = self.iris_dataset.iloc[:, 2:]

        plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=self.iris_target, cmap='coolwarm')
        plt.xlabel(f'{iris_part} len')
        plt.ylabel(f'{iris_part} width')
        plt.title(f'{iris_part} width & len correlation to Iris Type')
        plt.show()

    def visualize_heatmap(self) -> None:
        """
        Method to present the correlation of sepals or petals as a heatmap
        Utilizes seaborn package
        """
        sns.heatmap(self.iris_dataset.corr())
        plt.title('Correlation on iris classes')
        plt.show()

    def classify_data(self) -> None:
        """
        Method to create classifier
        """
        """
        train_test_split is a function in Sklearn model selection for splitting data arrays into two subsets: 
        for training data and for testing data. With this function, you don't need to divide the dataset manually.
        By default, Sklearn train_test_split will make random partitions for the two subsets. 
        However, you can also specify a random state for the operation.
        """
        x_train, x_test, y_train, y_test = train_test_split(self.iris_dataset.iloc[:, :-1], self.iris_target, test_size=0.25)

        # linear kernel is used to classify data
        classifier = SVC(kernel='linear')

        # fit the SVM model according to the given training data.
        classifier.fit(x_train, y_train)

        # perform classification on X, X being all the data from the dataset, without the type of Iris
        y_pred = classifier.predict(x_test)

        # confusion matrix is needed to predict the accuracy of classification
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=20)

        # {:.2f} is used to reduce the number of decimal points to 2
        print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
        print("Std Deviation: {:.2f} %".format(accuracies.std()*100))

    def run(self) -> None:
        """
        Method to run the program
        """
        # self.visualize_data("Sepal")
        # self.visualize_data("Petal")
        # self.visualize_heatmap()
        self.classify_data()


if __name__ == '__main__':
    dataset = IrisSetup().create_dataset_from_file()
    IrisClassification(dataset).run()
