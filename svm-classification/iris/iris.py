"""
Iris Petal SVM Classification
Authors: Wojciech Kudłacik, Norbert Daniluk
This program classifies data from the Iris Dataset.
Type of an Iris flower is being determined by the length and width of its petals and sepals.
Link: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
It utilizes sklearn library as the primary engine to perform classification.
Tutorials used:
- https://medium.com/@pinnzonandres/iris-classification-with-svm-on-python-c1b6e833522c
- https://dataaspirant.com/svm-classifier-implemenation-python-scikit-learn/?fbclid=IwAR1l-T9EQAy3bxPVfudCby0Qmlvd6AnqRBioLALRTVHS_HK5yj4b5u5Law0
"""

from iris_setup import IrisSetup
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC


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
    def get_user_input() -> dict:
        """
        Method to get input from a user to adjust classification parameters
        :return: Dictionary of parameters to adjust the algorithm
        """
        test_size = input("Enter test size value (0.25 default): ")
        if test_size is "":
            test_size = 0.25
            print(test_size)

        tts_rs = input("Enter random state seed value for tts method: ")
        if tts_rs is "":
            tts_rs = np.random.randint(100)
            print(tts_rs)
        else:
            tts_rs = int(tts_rs)

        ker_rs = input("Enter random state seed value for kernel: ")
        if ker_rs is "":
            ker_rs = np.random.randint(100)
            print(ker_rs)
        else:
            ker_rs = int(ker_rs)

        cv = input("Enter number of folds value (default 5): ")
        if cv is "":
            cv = "5"

        inputs = {
            "test_size": test_size,
            "tts_rs": tts_rs,
            "ker_rs": ker_rs,
            "cv": cv
        }

        return inputs

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

    def classify_data(self, test_size: int, tts_rs: int, ker_rs: int, cv: int) -> None:
        """
        Method to predict the outcome of classification. Dataset is split into test data and train subsets.
        Method prints out the classification report, accuracy of the training and standard deviation.
        :param test_size: How much of the dataset should be split as test data
        :param tts_rs: Seed of the data randomizer (by default it is a np.rand value)
        :param ker_rs: Seed of the kernel randomizer (by default it is a np.rand value)
        :param cv: Determines the cross-validation splitting strategy. Possible inputs for cv are:
                - None, to use the default 5-fold cross validation, - int, to specify the number of folds
        """
        """
        train_test_split is a function in Sklearn model selection for splitting data arrays into two subsets: 
        for training data and for testing data. With this function, you don't need to divide the dataset manually.
        By default, Sklearn train_test_split will make random partitions for the two subsets. 
        However, you can also specify a random state for the operation.
        """
        x_train, x_test, y_train, y_test = train_test_split(self.iris_dataset.iloc[:, :-1], self.iris_target,
                                                            test_size=float(test_size), random_state=tts_rs)
        # linear kernel is used to classify data
        classifier = SVC(kernel='linear', random_state=ker_rs)
        # fit the SVM model according to the given training data.
        classifier.fit(x_train, y_train)
        # perform classification on X, X being all the data from the dataset, without the type of Iris
        y_pred = classifier.predict(x_test)
        accuracies = cross_val_score(classifier, x_train, y_train, cv=int(cv))
        print(classification_report(y_test, y_pred))
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
        inputs = self.get_user_input()
        test_size = inputs.get("test_size")
        tts_rs = inputs.get("tts_rs")
        ker_rs = inputs.get("ker_ts")
        cv = inputs.get("cv")

        self.classify_data(test_size, tts_rs, ker_rs, cv)


if __name__ == '__main__':
    dataset = IrisSetup().create_dataset_from_file()
    IrisClassification(dataset).run()

    # Methods to use if you want to get results from User input
    # @staticmethod
    # def get_user_input() -> List[str]:
    #     """
    #     Method to get input from User
    #     :return: List of user inputs, that describe Iris flower
    #     """
    #     sepal_len = input("Sepal Len [cm]: ")
    #     sepal_width = input("Sepal Width [cm]: ")
    #     petal_len = input("Petal Len [cm]: ")
    #     petal_width = input("Sepal Width [cm]: ")
    #     return [sepal_len, sepal_width, petal_len, petal_width]

    #
    # def train(self) -> SVC:
    #     """
    #     Method to train the classifier based on the provided dataset
    #     :return: classifier: SVC model
    #     """
    #     x_train, x_test, y_train, y_test = train_test_split(self.iris_measurements, self.iris_target, test_size=0.25)
    #     classifier = SVC(kernel='linear')
    #     classifier.fit(x_train, y_train)
    #     return classifier
    #
    # @staticmethod
    # def make_prediction(classifier, measurements) -> str:
    #     """
    #     Makes a prediction based on the previously learnt data and provided input from a user
    #     :param classifier: SVC model
    #     :param measurements: User input that describes Iris flower
    #     :return: Predicted Iris Flower class
    #     """
    #     return classifier.predict([measurements])

