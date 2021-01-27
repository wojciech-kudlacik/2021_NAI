import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np


class NeuralClassification:

    def __init__(self):
        self.dataset = ''
        self.dataset_name = ''
        self.categories = ''

    def get_user_input(self):
        dataset_number = input('Choose dataset (1-4): ')
        if 1 <= int(dataset_number) <= 4:
            self.dataset = self.choose_dataset(int(dataset_number))
            (x_train, y_train), (x_test, y_test) = self.unpack_dataset(self.dataset)

            new_model = input('Train new model? (y/n): ')
            if new_model == 'y':
                epochs = input('Number of epochs?: ')
                model = self.train(x_train, y_train, x_test, y_test, epochs, dataset_number)

                save_model = input('Save model? (input name if you want to save it): ')
                if save_model != "":
                    self.save_model(model[0], save_model)
                else:
                    print('Model not saved')

                self.predict_from_dataset(model[0], x_test, y_test)

            else:
                model_path = input('Specify model path: ')
                model = self.read_model_from_file(model_path)
                self.predict_from_dataset(model, x_test, y_test)

        else:
            print('Out of range. Try again')

    @staticmethod
    def unpack_dataset(dataset):
        return dataset.load_data()

    def train(self, x_train, y_train, x_test, y_test, epochs, dataset_number):

        normalized_x_train = tf.keras.utils.normalize(x_train, axis=1)
        normalized_x_test = tf.keras.utils.normalize(x_test, axis=1)

        layers = {'1': self.iris_layers() ,'2': self.cifar_layers(), '3': self.fashion_layers(), '4': self.heritage_layers()}
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

    def heritage_layers(self):
        pass

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
            # load external dataset

        return dataset

    @staticmethod
    def evaluate_model(model, x_test, y_test):
        val_loss, val_acc = model.evaluate(x_test, y_test)
        print('Accuracy: {:.2f}'.format(val_acc))
        print('Loss: {:.2f}'.format(val_loss))

    @staticmethod
    def save_model(model, path):
        model.save(path + '.model')
        print('Saved')

    def predict_from_dataset(self, model, x_test, y_test):
        prediction = model.predict([x_test])
        predicted_category = np.argmax(prediction[123])
        print(predicted_category)
        print('Predicted category: ' + self.categories[int(predicted_category)])

        plt.imshow(x_test[123])
        print('Actual category: ' + self.categories[int(y_test[123])])
        plt.show()

    @staticmethod
    def read_model_from_file(path):
        return tf.keras.models.load_model(path + '.model')

    def run(self):
        self.get_user_input()


if __name__ == '__main__':
    NeuralClassification().run()
