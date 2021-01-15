"""
Iris Petal SVM Classification
Authors: Wojciech Kudłacik, Norbert Daniluk
"""
import pandas as pd


class IrisSetup:
    """
    Class to setup the data for the SVM program
    """
    @staticmethod
    def create_dataset_from_file():
        column_names = ["sepal_length_in_cm", "sepal_width_in_cm", "petal_length_in_cm", "petal_width_in_cm", "class"]
        iris_dataset = pd.read_csv("iris.data", header=None, names=column_names)
        return iris_dataset