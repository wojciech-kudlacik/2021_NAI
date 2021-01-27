import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt


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
                model = self.train(x_train, y_train, x_test, y_test)

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

    def train(self, x_train, y_train, x_test, y_test):

        normalized_x_train = tf.keras.utils.normalize(x_train, axis=1)
        normalized_x_test = tf.keras.utils.normalize(x_test, axis=1)

        model = Sequential()
        # model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
        #                  input_shape=(32, 32, 1)))
        # model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        # model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.fit(normalized_x_train, y_train, epochs=1)

        self.evaluate_model(model, normalized_x_test, y_test)

        return [model, normalized_x_test, y_test]

    def choose_dataset(self, dataset_number):
        dataset = None
        if dataset_number == 1:
            print('iris dataset')
            self.dataset_name = 'Iris'
        elif dataset_number == 2:
            print('CIFAR 10')
            dataset = tf.keras.datasets.cifar10
            self.dataset_name = 'cifar'
        elif dataset_number == 3:
            print('Fashion MNIST')
            dataset = tf.keras.datasets.fashion_mnist
            self.dataset_name = 'fashion'
        elif dataset_number == 4:
            print('MNIST')
            dataset = tf.keras.datasets.reuters

        return dataset

    def choose_categories(self):
        if self.dataset_name == 'cifar':
            self.categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.dataset_name == 'fashion':
            self.categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        elif self.dataset_name == ''


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
        self.choose_categories()
        print(self.categories)
        model.predict([x_test])
        print(y_test[123])
        plt.imshow(x_test[123])
        print(self.categories[int(y_test[123][0])])
        plt.show()

    @staticmethod
    def read_model_from_file(path):
        return tf.keras.models.load_model(path)

    def run(self):
        self.get_user_input()


if __name__ == '__main__':
    NeuralClassification().run()
