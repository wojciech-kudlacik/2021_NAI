"""
Neural Network (mostly) Image Classifier
Authors: Wojciech Kudłacik, Norbert Daniluk
This program classifies data from 4 different datasets:
- Iris Flowers dataset
- CIFAR10
- Fashion MNIST
- Architectural Heritage
It utilizes tensorflow and keras as the primary engine for classification
Tutorials used: https://www.youtube.com/watch?v=wQ8BIBpya2k&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN&ab_channel=sentdex
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class NeuralClassification:
    """
    Main class to perform neural network training and classification
    """
    def __init__(self):
        self.dataset = ''
        self.dataset_name = ''
        self.categories = ''

    def get_user_input(self):
        """
        Method to create 'menu'.
        Asks user for various inputs (more info inside README.md) and acts accordingly to them
        """
        dataset_number = input('Choose dataset (1-4): ')
        if 1 <= int(dataset_number) <= 4:
            self.dataset = self.choose_dataset(int(dataset_number))

            if dataset_number == '1':
                (x_train, y_train), (x_test, y_test) = self.load_iris_data()
            elif dataset_number == '2':
                (x_train, y_train), (x_test, y_test) = self.unpack_dataset(self.dataset)
            elif dataset_number == '3':
                (x_train, y_train), (x_test, y_test) = self.unpack_dataset(self.dataset)
            elif dataset_number == '4':
                (x_train, y_train), (x_test, y_test) = self.load_heritage_data()

            new_model = input('Train new model? (y/n): ')
            if new_model == 'y':
                epochs = input('Number of epochs?: ')
                model = self.train(x_train, y_train, x_test, y_test, epochs, dataset_number)

                save_model = input('Save model? (input name if you want to save it): ')
                if save_model != "":
                    self.save_model(model[0], save_model)
                else:
                    print('Model not saved')

                index = input('Index of a feature to test: ')
                self.predict_from_dataset(model[0], x_test, y_test, index)

            else:
                model_path = input('Specify model path: ')
                index = input('Index of a picture to test: ')
                model = self.read_model_from_file(model_path)
                self.predict_from_dataset(model, x_test, y_test, index)

        else:
            print('Out of range. Try again')

    def choose_dataset(self, dataset_number):
        """
        :param dataset_number: number taken from the user input
        :return dataset: either a tf dataset, or svm dataset or custom one
        """
        dataset = None
        if dataset_number == 1:
            print('iris dataset')
            self.categories = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
            dataset = self.load_iris_data()
        elif dataset_number == 2:
            print('CIFAR 10')
            self.categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            dataset = tf.keras.datasets.cifar10
        elif dataset_number == 3:
            print('Fashion MNIST')
            self.categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag',
                'Ankle boot']
            dataset = tf.keras.datasets.fashion_mnist
        elif dataset_number == 4:
            print('Architectural Heritage')
            self.categories = ['Altar', 'Apse', 'Bell tower', 'Column', 'Dome (inner)', 'Dome (outer)', 'Flying buttress',
                               'Gargoyle (and Chimera)', 'Stained glass', 'Vault']
            dataset = self.load_heritage_data()

        return dataset

    @staticmethod
    def unpack_dataset(dataset):
        """
        Method to unpack dataset from tensorflow
        :param dataset: dataset from tensorflow library to unpack
        :return: dataset loaded from tensorflow
        """
        return dataset.load_data()

    def train(self, x_train, y_train, x_test, y_test, epochs, dataset_number):
        """
        Method to train the model
        :param x_train: values to train
        :param y_train: labels to train
        :param x_test: values to test
        :param y_test: labels to test
        :param epochs: number of epochs (taken from input)
        :param dataset_number: number of the dataset (taken from input)
        :return: array that contains model, and x values and y labels
        """
        # hack for the dataset loaded from pickle files
        if dataset_number == '4':
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_test = np.array(x_test)
            y_test = np.array(y_test)

        normalized_x_train = tf.keras.utils.normalize(x_train, axis=1)
        normalized_x_test = tf.keras.utils.normalize(x_test, axis=1)

        layers = {'1': self.iris_layers(), '2': self.cifar_layers(), '3': self.fashion_layers(), '4': self.heritage_layers()}
        model = layers.get(dataset_number)

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(normalized_x_train, y_train, epochs=int(epochs))

        self.evaluate_model(model, normalized_x_test, y_test)

        return [model, normalized_x_test, y_test]

    @staticmethod
    def iris_layers():
        """
        Method to create layers for the iriS dataset
        :return: model to be compiled
        """
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3))
        model.add(Activation('softmax'))

        return model

    @staticmethod
    def cifar_layers():
        """
        Method to create layers for the CIFAR dataset
        :return: model to be compiled
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))

        return model

    @staticmethod
    def fashion_layers():
        """
        Method to create layers for the Fashion MNIST dataset
        :return: model to be compiled
        """
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))

        return model

    @staticmethod
    def heritage_layers():
        """
        Method to create layers for the heritage dataset
        :return: model to be compiled
        """
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=(128, 128, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(10))
        model.add(Activation('sigmoid'))

        return model

    @staticmethod
    def evaluate_model(model, x_test, y_test):
        """
        Method to predict accuracy and loss of the model
        :param model: trained model
        :param x_test: values to test
        :param y_test: labels to test
        """
        val_loss, val_acc = model.evaluate(x_test, y_test)
        print('Accuracy: {:.2f}'.format(val_acc))
        print('Loss: {:.2f}'.format(val_loss))

    def predict_from_dataset(self, model, x_test, y_test, index):
        """
        Method to predict the outcome. Uses test datasets.
        :param model: trained model
        :param x_test: values to test
        :param y_test: labels to test
        :param index: index of the test item that you want to check
        """
        prediction = model.predict([x_test])

        predicted_category = np.argmax(prediction[int(index)])
        print('Predicted category: ' + self.categories[int(predicted_category)])

        if 'Iris-setosa' in self.categories:
            print('Actual category: ' + self.categories[int(y_test[int(index)])])
        else:
            plt.imshow(x_test[int(index)])
            print('Actual category: ' + self.categories[int(y_test[int(index)])])
            plt.show()

    @staticmethod
    def load_heritage_data():
        """
        Method to load heritage data from files
        :return: train and test tuples created from the pickle files
        """
        x_train = pickle.load(open('x_train.pickle', 'rb'))
        y_train = pickle.load(open('y_train.pickle', 'rb'))
        x_test = pickle.load(open('x_test.pickle', 'rb'))
        y_test = pickle.load(open('y_test.pickle', 'rb'))

        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def load_iris_data():
        """
        Method to load iris dataset from scikit
        :return: train and test tuples
        """
        iris_data = load_iris()
        x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.1)
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def save_model(model, path):
        """
        Method to save model
        :param model: trained model
        :param path: path to save the file
        """
        model.save(path + '.model')
        print('Saved')

    @staticmethod
    def read_model_from_file(path):
        """
        Method to read model from file
        :param path: path to the model
        """
        return tf.keras.models.load_model(path + '.model')

    def run(self):
        """
        Method to run the program
        """
        self.get_user_input()


if __name__ == '__main__':
    NeuralClassification().run()
