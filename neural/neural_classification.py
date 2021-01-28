"""
Neural Network (mostly) Image Classifier
Authors: Wojciech Kudłacik, Norbert Daniluk
This program classifies data from 4 different datasets:
- Iris Flowers dataset
- CIFAR10
- Fashion MNIST
- Architectural Heritage
It utilizes tensorflow and keras as the primary engine for classification
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation
import matplotlib.pyplot as plt
import numpy as np
import pickle


class NeuralClassification:

    def __init__(self):
        self.dataset = ''
        self.dataset_name = ''
        self.categories = ''

    def get_user_input(self):
        dataset_number = input('Choose dataset (1-4): ')
        if 1 <= int(dataset_number) <= 4:
            self.dataset = self.choose_dataset(int(dataset_number))

            if dataset_number == '1':
                pass
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

                index = input('Index of a picture to test: ')
                self.predict_from_dataset(model[0], x_test, y_test, index)

            else:
                model_path = input('Specify model path: ')
                index = input('Index of a picture to test: ')
                model = self.read_model_from_file(model_path)
                self.predict_from_dataset(model, x_test, y_test, index)

        else:
            print('Out of range. Try again')

    @staticmethod
    def unpack_dataset(dataset):
        return dataset.load_data()

    def train(self, x_train, y_train, x_test, y_test, epochs, dataset_number):
        print(dataset_number)

        if dataset_number == '4':
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_test = np.array(x_test)
            y_test = np.array(y_test)

        normalized_x_train = tf.keras.utils.normalize(x_train, axis=1)
        normalized_x_test = tf.keras.utils.normalize(x_test, axis=1)

        layers = {'1': self.iris_layers(), '2': self.cifar_layers(), '3': self.fashion_layers(), '4': self.heritage_layers(x_train)}
        model = layers.get(dataset_number)

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(normalized_x_train, y_train, epochs=int(epochs))

        self.evaluate_model(model, normalized_x_test, y_test)

        return [model, normalized_x_test, y_test]

    @staticmethod
    def cifar_layers():
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
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))

        return model

    @staticmethod
    def heritage_layers(x_train):
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

    def iris_layers(self):
        pass

    def choose_dataset(self, dataset_number):
        dataset = None
        if dataset_number == 1:
            print('iris dataset')
            self.categories = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
            # load external dataset
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
    def load_heritage_data():
        x_train = pickle.load(open('x_train.pickle', 'rb'))
        y_train = pickle.load(open('y_train.pickle', 'rb'))
        x_test = pickle.load(open('x_test.pickle', 'rb'))
        y_test = pickle.load(open('y_test.pickle', 'rb'))

        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def evaluate_model(model, x_test, y_test):
        val_loss, val_acc = model.evaluate(x_test, y_test)
        print('Accuracy: {:.2f}'.format(val_acc))
        print('Loss: {:.2f}'.format(val_loss))

    @staticmethod
    def save_model(model, path):
        model.save(path + '.model')
        print('Saved')

    def predict_from_dataset(self, model, x_test, y_test, index):
        prediction = model.predict([x_test])
        predicted_category = np.argmax(prediction[int(index)])
        print(predicted_category)
        print('Predicted category: ' + self.categories[int(predicted_category)])

        plt.imshow(x_test[int(index)])
        print('Actual category: ' + self.categories[int(y_test[int(index)])])
        plt.show()

    @staticmethod
    def read_model_from_file(path):
        return tf.keras.models.load_model(path + '.model')

    def run(self):
        self.get_user_input()


if __name__ == '__main__':
    NeuralClassification().run()
