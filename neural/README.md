- [Assignment 6 - Data Classification with Neural Networks](#assignment-6---data-classification-with-neural-networks)
  * [Setup](#setup)
  * [About the program](#about-the-program)
  * [How to run](#how-to-run)
  * [Datasets](#datasets)
    + [Iris flowers dataset](#iris-flowers-dataset)
    + [CIFAR10](#cifar10)
    + [Fashion MNIST](#fashion-mnist)
    + [Architectural Heritage](#architectural-heritage)
  * [Useful Links](#useful-links)

# Assignment 6 - Data Classification with Neural Networks
The goal of this assignment was to implement a data classification algorithm with the assistance of neural networks using four different datasets.
Main engine for this exercise was [Tensorflow](https://www.tensorflow.org/) + [Keras](https://keras.io/). 
The solution was to be documented using **Docstring**.

## Setup
**Important**

The solution uses **Tensorflow v.2.4.1** that requires python to be of versions **between 3.5 to 3.8**.
The following solution was built using **python 3.8**.

In order to run the solution you need to have the following dependencies installed: 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all of them separately.

Alternatively, you can run:
```bash
pip install -r requirements.txt
```
to install all of the required packages for all exercises project

### Installing scikit-learn
```bash
pip install scikit-learn
```

### Installing numpy
```bash
pip install numpy
```

### Installing tensorflow
```bash
pip install tensorflow
```

### Installing matplotlib
```bash
pip install matplotlib
```

### Installing opencv
```bash
pip install opencv-python
```

## About the program
In this project, there are 4 different datasets being classified:
- [Iris flowers dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [Architectural Heritage](https://old.datahub.io/dataset/architectural-heritage-elements-image-dataset#:~:text=Architectural%20Heritage%20Elements%20Dataset%20(AHE,classification%20of%20architectural%20heritage%20images.&text=Most%20of%20the%20images%20have,them%20under%20creative%20commons%20license).)

## How to run

**IMPORTANT**

If you want to use load in Heritage Dataset you first have to download it from here (it's the first one from the three) [Architectural Heritage](https://old.datahub.io/dataset/architectural-heritage-elements-image-dataset#:~:text=Architectural%20Heritage%20Elements%20Dataset%20(AHE,classification%20of%20architectural%20heritage%20images.&text=Most%20of%20the%20images%20have,them%20under%20creative%20commons%20license).)

Unzip it, copy the files into this path:

```
path/to/project/neural
```

and rename it into

```
heritage_dataset
```

then, while still being in the **neural** directory use this command:

```bash
python load_data_set_from_file.py
```

This will create the needed files for training and testing purposes  

---

In order to run the program use this command:
```bash
cd into/project/path/neural
python3 neural_classification.py
``` 

You will the be asked to input a few parameters:

**Choose dataset (1-4)** - choose the dataset you want to train data for / or predict results.
- 1: Iris Flowers
- 2: CIFAR 10
- 3: Fashion MNIST
- 4: Architectural Heritage

**Train new model? (y/n)** - choose **y** if you want to train model from scratch or **n** if you want to use an existing model

If you have chosen to create a new model:

**Number of epochs?** - choose the number of epochs (basically how many times you want the network to run)

**Save model?** - input name / path if you want to save it (if you leave it blank, meaning you just press enter, the model won't be saved), .model is automatically added at the end (also I recommend using models/your_model as input)

If you have chosen to use an existing model:

**Specify model path** - specify path to the existing model (most likely models/your_model)

In both cases you will then see this message:

**Index of a feature / picture to test** - choose index of the data you want to test your model against

## Datasets

### Iris flowers dataset
The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician, eugenicist, and biologist Ronald Fisher in his 1936 paper *The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis*. Based on Fisher's linear discriminant model, this data set became a typical test case for many statistical classification techniques in machine learning such as support vector machines.

The program classifies Iris flowers and assigns them a class (iris-setosa, iris-versicolor or iris-virginica) based on the length and width of petals and sepals. 

### Outputs

Two models created 

### 100 epochs

#### Accuray and loss
![Imgur](https://i.imgur.com/9YXUKhI.png)

#### Data test
![Imgur](https://i.imgur.com/5fhxUzW.png)

### 150 epochs

#### Accuray and loss
![Imgur](https://i.imgur.com/BWdd2zu.png)

#### Data test
![Imgur](https://i.imgur.com/5fhxUzW.png)

### Comparison with SVM

After 150 epochs, the accuracy seems to be similar. 

![Imgur](https://i.imgur.com/3g7Mkzk.png)

### CIFAR10
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

### Outputs

Two models created

### 10 epochs

#### Accuracy and loss
![Imgur](https://i.imgur.com/FXG3UDq.png)

#### Image test
![Imgur](https://i.imgur.com/83FL7Yr.png)

### 20 epochs

#### Accuracy and loss
![Imgur](https://i.imgur.com/2HEivrY.png)

#### Image test
![Imgur](https://i.imgur.com/Ni3UuBT.png)

### Fashion MNIST
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

### Outputs

Two models created

### 30 epochs

#### Accuracy and loss
![Imgur](https://i.imgur.com/bYYsTLb.png)

#### Image test
![Imgur](https://i.imgur.com/jHQYGlv.png)

### 50 epochs

#### Accuracy and loss
![Imgur](https://i.imgur.com/GeQ3sdH.png)

#### Image test
![Imgur](https://i.imgur.com/i9mQh84.png)

### Architectural Heritage
Architectural Heritage Elements Dataset (AHE) is an image dataset for developing deep learning algorithms and specific techniques in the classification of architectural heritage images. This dataset consists of 10235 images classified in 10 categories: Altar: 829 images; Apse: 514 images; Bell tower: 1059 images; Column: 1919 images; Dome (inner): 616 images; Dome (outer): 1177 images; Flying buttress: 407 images; Gargoyle (and Chimera): 1571 images; Stained glass: 1033 images; Vault: 1110 images. It is inspired by the CIFAR-10 dataset but with the objective in mind of developing tools that facilitate the tasks of classifying images in the field of cultural heritage documentation. Most of the images have been obtained from Flickr and Wikimedia Commons (all of them under creative commons license).

### Outputs

Two models created

### 10 epochs

#### Accuracy and loss
![Imgur](https://i.imgur.com/3C9fy7R.png)

#### Image test
![Imgur](https://i.imgur.com/ANlN2rG.png)

### 20 epochs

#### Accuracy and loss
![Imgur](https://i.imgur.com/F0JFE0T.png)

#### Image test
![Imgur](https://i.imgur.com/bvlyc6T.png)

## Useful Links

### Docstring
[Python Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)

[Python Docstring Tutorial](https://www.datacamp.com/community/tutorials/docstrings-python)

### scikit-learn
[scikit-learn](https://scikit-learn.org/stable/)

[scikit-learn svm](https://scikit-learn.org/stable/modules/svm.html)

### Tensorflow
[Tensorflow](https://www.tensorflow.org/)

### Keras
[Keras](https://keras.io/)

### Datasets
- [Iris flowers dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [Architectural Heritage](https://old.datahub.io/dataset/architectural-heritage-elements-image-dataset#:~:text=Architectural%20Heritage%20Elements%20Dataset%20(AHE,classification%20of%20architectural%20heritage%20images.&text=Most%20of%20the%20images%20have,them%20under%20creative%20commons%20license).)


