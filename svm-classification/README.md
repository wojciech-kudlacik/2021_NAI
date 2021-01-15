# Assignment 5 - Data Classification
The goal of this assignment was to implement a data classification algorithm using two different datasets and scikit-learn svm library with additional use of pandas and plotting libraries. The solution was to be documented using **Docstring**.

## Setup
In order to run the solution you need to have the following dependencies installed: 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all of them.

### Installing scikit-learn
```bash
pip install scikit-learn
```

### Installing numpy
```bash
pip install numpy
```

### Installing pandas
```bash
pip instal pandas
```

### Installing seaborn
```bash
pip install seaborn
```

### Installing matplotlib
```bash
pip install matplotlib
```

## About the program
In this project, there are two different datasets being classified:
- Iris flowers dataset
- SOCR data of height and weight of people

### Iris flowers dataset
The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician, eugenicist, and biologist Ronald Fisher in his 1936 paper *The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis*. Based on Fisher's linear discriminant model, this data set became a typical test case for many statistical classification techniques in machine learning such as support vector machines.

The program classifies Iris flowers and assigns them a class (iris-setosa, iris-versicolor or iris-virginica) based on the length and width of petals and sepals. 

### How to run
```bash
cd into/project/path/svm-classification/iris
python iris.py
```
You will be then asked to input parameters for the to adjust the underlying classifier.
You will need to provide:
- **test_size**: value from 0.1 to 0.99 (defaults to 0.25)
- **tts_rs**: splitting method random seed (to randomize data each time), if empty a random int is generated.
- **ker_rs**: kernel random seed (to randomize data each time), if empty a random int is generated.
- **cv**: int, to specify the number of folds

The result will be a matrix of predicted classes for the Iris flowers in the test subset of the Iris dataset.

### Outputs

Predicted accuracy and standard deviation is quite high, due to the quality of the dataset.

![Imgur](https://i.imgur.com/FiRwJc2.png)

![Imgur](https://i.imgur.com/dcQhLY7.png)

![Imgur](https://i.imgur.com/HJGS5z6.png)

### Data visualization
![Imgur](https://i.imgur.com/xx1wdOK.png)

![Imgur](https://i.imgur.com/TIrO1Kw.png)

![Imgur](https://i.imgur.com/LvkFOzD.png)

### SOCR data

Human Height and Weight are mostly hereditary, but lifestyles, diet, health and environmental factors also play a role in determining individual's physical characteristics. The dataset below contains 25,000 synthetic records of human heights and weights of 18 years old children. These data were simulated based on a 1993 by a Growth Survey of 25,000 children from birth to 18 years of age recruited from Maternal and Child Health Centres (MCHC) and schools and were used to develop Hong Kong's current growth charts for weight, height, weight-for-age, weight-for-height and body mass index (BMI).

### How to run
```bash
cd into/project/path/svm-classification/measurements
python measurements.py
```
You will then be asked to provide two inputs:
- Height (in inches)
- Weight (in pounds)

Based on the underlying calculations, the outcome will be the gender of the person based on their height and weight.
 
### Outputs

![Imgur](https://i.imgur.com/6STboRR.png)

![Imgur](https://i.imgur.com/x7ksB3U.png)

![Imgur](https://i.imgur.com/7i8vzIu.png)

## Useful Links

### Docstring
[Python Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)
[Python Docstring Tutorial](https://www.datacamp.com/community/tutorials/docstrings-python)

### scikit-learn
[scikit-learn](https://scikit-learn.org/stable/)
[scikit-learn svm](https://scikit-learn.org/stable/modules/svm.html)

### Datasets
[Iris flowers dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/)
[H&W dataset](https://www.kaggle.com/mustafaali96/weight-height)


