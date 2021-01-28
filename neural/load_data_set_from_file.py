"""
Neural Network (mostly) Image Classifier
Authors: Wojciech Kudłacik, Norbert Daniluk
This script loads Architectural Heritage, splits it into train and test data and saves to files.
Tutorials used: https://www.youtube.com/watch?v=wQ8BIBpya2k&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN&ab_channel=sentdex
"""
import numpy as np
import os
import cv2
import random
import pickle


class LoadDataset:
    """
    Class that contains methods to create files for the Architectural Heritage dataset
    """
    def __init__(self):
        self.datadir = './heritage_dataset'
        self.categories = ['altar', 'apse', 'bell_tower', 'column', 'dome(inner)', 'dome(outer)', 'flying_buttress',
                               'gargoyle', 'stained_glass', 'vault']
        self.data = []
        self.X = []
        self.y = []
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.filenames = ['x_train', 'y_train', 'x_test', 'y_test']

    def create_training_data(self):
        """
        This method iterates over files in the heritage_dataset directory,
        randomizes the order and creates X values and y labels
        """
        for category in self.categories:
            path = os.path.join(self.datadir, category)
            class_num = self.categories.index(category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    self.data.append([img_array, class_num])
                except Exception as e:
                    pass

        random.shuffle(self.data)

        for features, label in self.data:
            self.X.append(features)
            self.y.append(label)

        self.X = np.array(self.X).reshape(-1, 128, 128, 1)

    def split_dataset(self):
        """
        This method splits the X and y arrays into train and test arrays
        """
        self.x_train = self.X[:(len(self.X) // 10) * 8]
        self.y_train = self.y[:(len(self.y) // 10) * 8]
        self.x_test = self.X[(len(self.X) // 10) * 8:]
        self.y_test = self.y[(len(self.y) // 10) * 8:]

    def save_data(self):
        """
        This method saves the need train and test arrays into files to be used later
        """
        datasets = [self.x_train, self.y_train, self.x_test, self.y_test]
        for i, file_name in enumerate(self.filenames):
            pickle_out = open(file_name + '.pickle', 'wb')
            pickle.dump(datasets[i], pickle_out)
            pickle_out.close()

    def run(self):
        """
        Method to run the script
        """
        self.create_training_data()
        self.split_dataset()
        self.save_data()


if __name__ == '__main__':
    LoadDataset().run()
